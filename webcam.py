import cv2
import numpy as np

CAM_INDEX = 1  # try 0/1/2
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera not opened. Try CAM_INDEX = 0 or 2.")
    raise SystemExit

MIN_AREA = 800  # ignore small noise blobs

# HSV color ranges (tune if needed)
COLOR_RANGES = {
    "RED": [
        (np.array([0, 120, 80]),   np.array([10, 255, 255])),
        (np.array([170, 120, 80]), np.array([180, 255, 255]))
    ],
    "YELLOW": [
        (np.array([20, 120, 120]), np.array([35, 255, 255]))
    ],
    "GREEN": [
        (np.array([35, 80, 80]),   np.array([85, 255, 255]))
    ],
    "BLUE": [
        (np.array([95, 100, 70]),  np.array([135, 255, 255]))
    ]
}

def build_mask(hsv, ranges):
    mask = None
    for low, up in ranges:
        m = cv2.inRange(hsv, low, up)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask

kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    best_color = None
    best_contour = None
    best_area = 0
    best_mask = None

    # Find the largest colored object among all 4 colors
    for color_name, ranges in COLOR_RANGES.items():
        mask = build_mask(hsv, ranges)

        # Clean noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > best_area:
            best_area = area
            best_color = color_name
            best_contour = c
            best_mask = mask

    # Draw result if large enough
    if best_contour is not None and best_area > MIN_AREA:
        x, y, w, h = cv2.boundingRect(best_contour)
        cx, cy = x + w // 2, y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
        cv2.putText(frame, f"{best_color}  Area:{int(best_area)}  Center:({cx},{cy})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show windows
    cv2.imshow("Frame", frame)
    if best_mask is not None:
        cv2.imshow("Mask (largest color detected)", best_mask)
    else:
        cv2.imshow("Mask (largest color detected)", np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
