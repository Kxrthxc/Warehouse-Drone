import cv2
import numpy as np
import serial
import time

# ----------------------------
# SETTINGS (EDIT THESE)
# ----------------------------
CAM_INDEX = 1                      # 0/1/2
COM_PORT  = "COM9"                  # change to your Arduino COM port
BAUD      = 115200

MARKER_LENGTH_M = 0.05              # marker size in meters (e.g., 5 cm => 0.05)
TARGET_IDS = [1, 2, 3, 4]           # marker IDs to follow in order (waypoints)

# Control gains (tune)
K_STEER = 0.9                       # steering gain from x-offset
K_FWD   = 0.8                       # forward gain from distance
MAX_PWM = 180                       # max motor PWM (0..255)
STOP_DIST_M = 0.25                  # stop this close to marker (meters)
SEARCH_PWM = 90                     # rotate when marker not visible

# ----------------------------
# CAMERA CALIBRATION (IMPORTANT)
# ----------------------------
# BEST: load real calibration from a file.
# Replace these with your camera calibration values (or load from .npz).
# If you don't have calibration, it will still "work" for centering,
# but distance (meters) will be unreliable (Uncertain).
cameraMatrix = np.array([[900, 0, 640],
                         [0, 900, 360],
                         [0,   0,   1]], dtype=np.float32)
distCoeffs = np.zeros((5, 1), dtype=np.float32)

# ----------------------------
# INIT
# ----------------------------
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("❌ Camera not opened. Try CAM_INDEX = 1 or 2.")

ser = serial.Serial(COM_PORT, BAUD, timeout=0.1)
time.sleep(2)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

current_target_idx = 0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def send_motor(l, r):
    # l,r: -255..255
    msg = f"{int(l)},{int(r)}\n"
    ser.write(msg.encode("utf-8"))

print("✅ Running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    cx_frame = w // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    target_id = TARGET_IDS[current_target_idx] if current_target_idx < len(TARGET_IDS) else None

    found_target = False

    if ids is not None and target_id is not None:
        ids_flat = ids.flatten()
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # If target ID is visible, use it
        if target_id in ids_flat:
            i = int(np.where(ids_flat == target_id)[0][0])
            c = corners[i]

            # Pose estimation => rvec, tvec
            rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
                c, MARKER_LENGTH_M, cameraMatrix, distCoeffs
            )
            rvec, tvec = rvecs[0], tvecs[0]  # tvec in meters if calibration is correct

            # Draw axes
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, MARKER_LENGTH_M * 0.5)

            # Marker center (pixels)
            pts = c[0].astype(int)
            mx = int(pts[:, 0].mean())
            my = int(pts[:, 1].mean())
            cv2.circle(frame, (mx, my), 6, (0, 0, 255), -1)

            # Control signals
            x_err_px = mx - cx_frame
            x_err = x_err_px / (w / 2)                  # normalize to approx [-1, +1]

            z_dist = float(tvec[0][2])                  # forward distance in meters (needs calibration)

            # Forward speed reduces as we get close
            fwd = K_FWD * (z_dist - STOP_DIST_M)
            steer = K_STEER * x_err

            # Convert to left/right motor PWM
            l = (fwd - steer) * MAX_PWM
            r = (fwd + steer) * MAX_PWM

            l = clamp(l, -MAX_PWM, MAX_PWM)
            r = clamp(r, -MAX_PWM, MAX_PWM)

            # If close enough => stop and advance to next marker
            if z_dist <= STOP_DIST_M:
                send_motor(0, 0)
                cv2.putText(frame, f"Reached ID {target_id}! Next...",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                time.sleep(0.6)
                current_target_idx += 1
            else:
                send_motor(l, r)

            cv2.putText(frame, f"Target ID: {target_id}  x_err:{x_err:.2f}  z:{z_dist:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            found_target = True

    # If not found, rotate slowly to search
    if (not found_target) and (target_id is not None):
        send_motor(-SEARCH_PWM, SEARCH_PWM)
        cv2.putText(frame, f"Searching for ID {target_id}...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # If all targets done => stop
    if current_target_idx >= len(TARGET_IDS):
        send_motor(0, 0)
        cv2.putText(frame, "All markers reached. STOP.",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ArUco GPS-Denied Navigation", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

send_motor(0, 0)
cap.release()
cv2.destroyAllWindows()
ser.close()
