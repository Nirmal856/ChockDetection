import cv2
import torch
import time
import datetime
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForVision2Seq

# Use MPS (Mac GPU) or CPU if needed
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load YOLO Model (trained for chock detection)
yolo_model = YOLO("/Users/nirmal-10730/PycharmProjects/BuildMoniter/src/data_gen/feature/runs/detect/train3/weights/best.pt")

# Load SmolVLM Model (Only if enabled)
USE_SMOLVLM = False  # Set to False to completely disable SmolVLM verification
if USE_SMOLVLM:
    model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model_vlm = AutoModelForVision2Seq.from_pretrained(model_name).to(device)

yolo_interval = 3
last_yolo_time = 0

validate_interval = 12 
validate_count = 0
yolo_validate_count = 0
validate_start_time = 0
validation_needed = False
smolvlm_failed = False
smolvlm_validation_needed = False

# Chock status tracking
current_chock_state = None

# Log file path
log_file = "chock_detection_log.txt"


def log_event(event_type, event, smolvlm_needed):
    """Logs chock ON/OFF event with timestamp and validation type."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    smolvlm_status = "Yes" if smolvlm_needed else "No"
    log_entry = f"{timestamp} - [{event_type}] {event} (SmolVLM Needed: {smolvlm_status})\n"

    with open(log_file, "a") as f:
        f.write(log_entry)

    print(f"Logged: {log_entry.strip()}")


def is_chock_smolvlm(image):
    """Use SmolVLM to verify if the detected object is a chock."""
    if not USE_SMOLVLM or image is None or image.size == 0:
        return False

    image_resized = cv2.resize(image, (224, 224))
    prompt = "<image> Is this object a chock, used to secure aircraft wheels?"

    inputs = processor(images=[image_resized], text=prompt, return_tensors="pt").to(device)
    output = model_vlm.generate(**inputs)
    response = processor.batch_decode(output, skip_special_tokens=True)[0]
    return "yes" in response.lower()


# Open live video
cap = cv2.VideoCapture("/Users/nirmal-10730/Downloads/test4.mp4")
detected_chocks = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Run YOLO every 30 seconds
    if current_time - last_yolo_time >= yolo_interval:
        last_yolo_time = current_time
        detected_chocks.clear()

        results = yolo_model.predict(frame)
        chock_detected = False

        for r in results:
            for box in r.boxes:
                if box.xyxy.shape[0] == 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])

                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue

                chock_img = frame[y1:y2, x1:x2]
                if chock_img.size == 0:
                    continue

                chock_detected = True if class_id == 1 else False
                label = "Chock On" if chock_detected else "Chock Off"
                color = (0, 0, 255) if chock_detected else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detected_chocks.append(chock_img)

        # Log YOLO detection count
        if chock_detected:
            yolo_validate_count += 1
        else:
            yolo_validate_count = 0

        # Detect state change (On/Off)
        if chock_detected and current_chock_state != "on":
            validation_needed = True
            smolvlm_failed = False
            smolvlm_validation_needed = USE_SMOLVLM
            validate_count = 0
            validate_start_time = current_time
            current_chock_state = "on"

        elif not chock_detected and current_chock_state != "off":
            validation_needed = True
            smolvlm_failed = False
            smolvlm_validation_needed = USE_SMOLVLM
            validate_count = 0
            validate_start_time = current_time
            current_chock_state = "off"

    # Validate state 5 times in 1 min before logging
    if validation_needed and detected_chocks and current_time - validate_start_time <= 60:
        if USE_SMOLVLM:
            if current_time - validate_start_time >= validate_interval * validate_count:
                for chock in detected_chocks:
                    if is_chock_smolvlm(chock):
                        validate_count += 1
                        break

            if validate_count < 5 and current_time - validate_start_time > 60:
                smolvlm_failed = True

            if validate_count >= 5:
                log_event("SmolVLM", f"First Chock {current_chock_state.upper()}", smolvlm_validation_needed)
                validation_needed = False

    # If SmolVLM is disabled or fails, use YOLO (validate 5 times in 1 min)
    if (not USE_SMOLVLM or smolvlm_failed) and yolo_validate_count >= 5:
        smolvlm_validation_needed = False
        log_event("Fallback", f"Using YOLO for Chock {current_chock_state.upper()}", smolvlm_validation_needed)
        validation_needed = False
        smolvlm_failed = False

    # Show the processed frame
    cv2.imshow("Chock Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    pass
