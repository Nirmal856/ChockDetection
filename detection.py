import cv2
import torch
import time
import datetime
from collections import deque
from ultralytics import YOLO

model_path = "/Users/nirmal-10730/PycharmProjects/BuildMoniter/src/data_gen/meta/best_25m.pt"
yolo_model = YOLO(model_path)

device = "mps" if torch.cuda.is_available() else "cpu"

log_file = "chock_detection_log.txt"

previous_state = None
bounding_boxes = deque(maxlen=10)

validate_count = 0
validation_needed = False
validation_start_time = 0
validation_threshold = 5
validation_gap = 2

first_chock_on_logged = False
first_chock_off_logged = False

video_path = "/Users/nirmal-10730/Downloads/test6.mov"
cap = cv2.VideoCapture(video_path)

def log_event(event_type, event, video_time):
    log_entry = f"{video_time} - [{event_type}] {event}\n"

    with open(log_file, "a") as f:
        f.write(log_entry)
        f.flush()

    print(f"Logged: {log_entry.strip()}")

def get_video_time(cap):
    millis = cap.get(cv2.CAP_PROP_POS_MSEC)
    seconds = int(millis / 1000)
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    video_time = get_video_time(cap)

    results = yolo_model.predict(frame, conf=0.5, iou=0.45)

    chock_on_count = 0
    chock_off_count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])

            if class_id == 0:
                chock_off_count += 1
            elif class_id == 1:
                chock_on_count += 1

            bounding_boxes.append((x1, y1, x2, y2, (0, 255, 0) if class_id == 0 else (0, 0, 255)))

    detected_state = None
    if chock_on_count > 0 and chock_off_count == 0:
        detected_state = "on"
    elif chock_off_count > 0 and chock_on_count == 0:
        detected_state = "off"

    if detected_state and detected_state != previous_state:
        if not validation_needed:
            validation_needed = True
            validation_start_time = time.time()
            validate_count = 0

        elif time.time() - validation_start_time >= validation_gap * validate_count:
            validate_count += 1

        if validate_count >= validation_threshold:
            if detected_state == "on" and not first_chock_on_logged:
                #log_event("Chock State", "First Chock Applied", video_time)
                first_chock_on_logged = True
                first_chock_off_logged = False

            elif detected_state == "off" and not first_chock_off_logged:
                #log_event("Chock State", "First Chock Unapplied", video_time)
                first_chock_off_logged = True
                first_chock_on_logged = False

            log_event("Chock State", "Chocks " + ("Applied" if detected_state == "on" else "Removed"), video_time)

            previous_state = detected_state
            validation_needed = False

    for x1, y1, x2, y2, color in bounding_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Chock Detection", frame)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.2f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


if __name__ == "__main__":
    pass
