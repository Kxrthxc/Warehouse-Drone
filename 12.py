import cv2
import numpy as np

# --- Configuration ---
URL = "http://10.30.78.48" 
ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# FRAME CENTER for 640x480
CENTER_X, CENTER_Y = 320, 240

cap = cv2.VideoCapture(URL + ":81/stream")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    height, width = frame.shape[:2]

    # Draw FIXED center crosshair (320,240) - RED
    cv2.circle(frame, (CENTER_X, CENTER_Y), 8, (0, 0, 255), -1)
    cv2.line(frame, (CENTER_X-20, CENTER_Y), (CENTER_X+20, CENTER_Y), (0, 0, 255), 2)
    cv2.line(frame, (CENTER_X, CENTER_Y-20), (CENTER_X, CENTER_Y+20), (0, 0, 255), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            
            # Marker center position
            marker_center = tuple(np.mean(corners[i][0], axis=0).astype(int))
            x_pos, y_pos = marker_center
            
            # DEVIATION from center (MAIN FEATURE)
            dx = x_pos - CENTER_X
            dy = y_pos - CENTER_Y
            
            # CLEAN DISPLAY: ID + POSITION + DEVIATION
            text = f"ID:{marker_id} ({x_pos},{y_pos}) DX:{dx:+4d} DY:{dy:+4d}"
            cv2.putText(frame, text, (x_pos - 90, y_pos - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Green marker crosshair
            cv2.circle(frame, marker_center, 6, (0, 255, 0), -1)
            cv2.line(frame, (x_pos-12, y_pos), (x_pos+12, y_pos), (0, 255, 0), 2)
            cv2.line(frame, (x_pos, y_pos-12), (x_pos, y_pos+12), (0, 255, 0), 2)

    # Status info
    cv2.putText(frame, f"640x480 Center({CENTER_X},{CENTER_Y})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("ArUco Center Deviation (DX,DY)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()