import cv2
import numpy as np
import math
import socket
import struct
import time

# ================================
# SETTINGS
# ================================
STREAM_URL = "http://10.35.170.148/stream"
ESP_IP     = "10.35.170.148"
ESP_PORT   = 14550

MARKER_LENGTH = 50.0
CALIB_FILE = "camera_calibration.npz"
ARUCO_DICT = cv2.aruco.DICT_6X6_250

SEND_HZ  = 50
SEND_DT  = 1.0 / SEND_HZ

PRINT_HZ = 10
PRINT_DT = 1.0 / PRINT_HZ

# ================================
# RC RANGE
# ================================
RC_MIN, RC_MID, RC_MAX = 1000, 1500, 2000

# ================================
# X PID -> ROLL (CH1)
# ================================
SETPOINT_X_MM = 0.0

Kp = 2.0       # us/mm
Ki = 0.0       # us/(mm*s)
Kd = 0.8       # us/(mm/s)

MAX_ROLL_US = 200     # limit stick 1500 ± 200 while testing
I_LIMIT = 200.0       # clamp integral (mm*s)

X_LPF_ALPHA = 0.35
X_REVERSE = True     # set True if roll moves opposite

# ================================
# PID STATE
# ================================
x_i = 0.0
prev_x = None
x_filt = None

# ================================
# UDP SOCKET
# ================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def clamp(v, lo=RC_MIN, hi=RC_MAX):
    return max(lo, min(hi, int(v)))

def send_rc(seq, roll, pitch, throttle, yaw):
    payload = b"RC" + struct.pack("<4H", roll, pitch, throttle, yaw) + struct.pack("<B", seq & 0xFF)
    csum = sum(payload) & 0xFF
    packet = payload + struct.pack("<B", csum)
    sock.sendto(packet, (ESP_IP, ESP_PORT))

# ================================
# LOAD CALIBRATION
# ================================
data = np.load(CALIB_FILE)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# ================================
# ARUCO INIT
# ================================
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ================================
# STREAM
# ================================
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("❌ Cannot open stream:", STREAM_URL)
    raise SystemExit

print("▶ Running: X/Y/Z + Angle print  +  X-PID -> ROLL (CH1)")
print("Stops sending UDP when marker is lost (manual fallback).")
print("Press 'q' to exit.\n")

seq = 0
last_send = time.time()
last_print = time.time()
marker_was_seen = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    marker_seen = (ids is not None and len(ids) > 0)

    # defaults
    roll_rc     = RC_MID
    pitch_rc    = RC_MID
    yaw_rc      = RC_MID
    throttle_rc = RC_MIN

    if marker_seen:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, camera_matrix, dist_coeffs
        )

        # first marker only
        marker_id = int(ids[0][0])
        tvec = tvecs[0][0]
        x_mm, y_mm, z_mm = float(tvec[0]), float(tvec[1]), float(tvec[2])

        # ===== 2D "ANGLE" (NOT EULER) =====
        # angle of the marker's top edge in image plane
        # corners[0] shape: (1,4,2) -> take [0] -> (4,2)
        pts = corners[0][0]
        (x0, y0) = pts[0]
        (x1, y1) = pts[1]
        angle_deg = math.degrees(math.atan2((y1 - y0), (x1 - x0)))  # -180..+180

        # ===== PRINT X Y Z ANGLE (terminal) =====
        now = time.time()
        if (now - last_print) >= PRINT_DT:
            last_print = now
            print(f"ID:{marker_id}  X:{x_mm:+.1f}  Y:{y_mm:+.1f}  Z:{z_mm:+.1f} mm  ANG:{angle_deg:+.1f} deg")

        # ===== FILTER X =====
        if x_filt is None:
            x_filt = x_mm
        else:
            x_filt = (X_LPF_ALPHA * x_mm) + ((1.0 - X_LPF_ALPHA) * x_filt)

        # ===== PID UPDATE @ SEND_DT =====
        if (now - last_send) >= SEND_DT:
            dt = now - last_send
            dt = max(0.01, min(dt, 0.05))  # clamp
            last_send = now

            err = SETPOINT_X_MM - x_filt

            x_i += err * dt
            x_i = max(-I_LIMIT, min(I_LIMIT, x_i))

            if prev_x is None:
                d_meas = 0.0
            else:
                d_meas = (x_filt - prev_x) / dt
            prev_x = x_filt

            # PID output (microseconds offset around 1500)
            u = (Kp * err) + (Ki * x_i) - (Kd * d_meas)
            u = max(-MAX_ROLL_US, min(MAX_ROLL_US, u))

            if X_REVERSE:
                u = -u

            roll_rc = clamp(RC_MID + u)

            # send ONLY when marker is seen
            send_rc(seq, roll_rc, pitch_rc, throttle_rc, yaw_rc)
            seq = (seq + 1) & 0xFF

        # overlay
        cv2.putText(frame,
                    f"X:{x_filt:+.1f}mm  ROLL_RC:{roll_rc}  ANG:{angle_deg:+.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

        marker_was_seen = True

    else:
        # marker lost -> stop sending -> manual fallback
        if marker_was_seen:
            x_i = 0.0
            prev_x = None
            x_filt = None
            marker_was_seen = False

        now = time.time()
        if (now - last_print) >= PRINT_DT:
            last_print = now
            print("NO MARKER (not sending UDP -> manual fallback)")

        cv2.putText(frame, "NO MARKER", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    cv2.imshow("Aruco X/Y/Z + Angle + Roll PID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
