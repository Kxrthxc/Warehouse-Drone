import cv2
import numpy as np
import math
import socket
import struct
import time

# ================================
# SETTINGS
# ================================
STREAM_URL = "http://10.30.78.48/stream"
ESP_IP     = "10.30.78.48"
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

# =========================================================
# X PID -> ROLL (CH1)   | setpoint: X=0
# Y PID -> PITCH (CH2)  | setpoint: Y=...
# ANG PID -> YAW (CH4)  | setpoint: ANG=0 deg  (2D angle from corners)
# =========================================================
SETPOINT_X_MM   = 0.0
SETPOINT_Y_MM   = -30.0
SETPOINT_ANG_DEG = 0.0

# ---- X PID (ROLL) ----
Kp_x = 2.0
Ki_x = 0.0
Kd_x = 0.8

MAX_ROLL_US = 200
I_LIMIT_X   = 200.0
X_LPF_ALPHA = 0.35
X_REVERSE   = True

# ---- Y PID (PITCH) ----
Kp_y = 2.0
Ki_y = 0.0
Kd_y = 0.8

MAX_PITCH_US = 200
I_LIMIT_Y    = 200.0
Y_LPF_ALPHA  = 0.35
Y_REVERSE    = False

# ---- ANG PID (YAW) ----
# NOTE: this uses 2D marker "angle" from image plane, not Euler.
Kp_a = 4.0
Ki_a = 0.0
Kd_a = 1.2

MAX_YAW_US  = 200
I_LIMIT_A   = 200.0
ANG_LPF_ALPHA = 0.35
ANG_REVERSE = True

# ================================
# PID STATE
# ================================
x_i = 0.0
prev_x = None
x_filt = None

y_i = 0.0
prev_y = None
y_filt = None

a_i = 0.0
prev_a = None
a_filt = None

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

def wrap_deg180(a):
    """Wrap any angle to [-180, +180)."""
    return (a + 180.0) % 360.0 - 180.0

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

print("▶ Running: X/Y/Z + ANG print  +  X->ROLL + Y->PITCH + ANG->YAW PID")
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
    throttle_rc = RC_MIN  # safety

    if marker_seen:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, camera_matrix, dist_coeffs
        )

        # first marker only
        marker_id = int(ids[0][0])
        tvec = tvecs[0][0]
        x_mm, y_mm, z_mm = float(tvec[0]), float(tvec[1]), float(tvec[2])

        # ===== 2D ANGLE (NOT EULER) =====
        pts = corners[0][0]          # (4,2)
        (x0, y0) = pts[0]
        (x1, y1) = pts[1]
        ang_deg = math.degrees(math.atan2((y1 - y0), (x1 - x0)))  # -180..+180
        ang_deg = wrap_deg180(ang_deg)

        now = time.time()

        # ===== PRINT =====
        if (now - last_print) >= PRINT_DT:
            last_print = now
            print(f"ID:{marker_id}  X:{x_mm:+.1f}  Y:{y_mm:+.1f}  Z:{z_mm:+.1f} mm  ANG:{ang_deg:+.1f} deg")

        # ===== FILTERS =====
        if x_filt is None: x_filt = x_mm
        else: x_filt = (X_LPF_ALPHA * x_mm) + ((1.0 - X_LPF_ALPHA) * x_filt)

        if y_filt is None: y_filt = y_mm
        else: y_filt = (Y_LPF_ALPHA * y_mm) + ((1.0 - Y_LPF_ALPHA) * y_filt)

        if a_filt is None: a_filt = ang_deg
        else: a_filt = (ANG_LPF_ALPHA * ang_deg) + ((1.0 - ANG_LPF_ALPHA) * a_filt)
        a_filt = wrap_deg180(a_filt)

        # ===== PID UPDATE @ SEND_DT =====
        if (now - last_send) >= SEND_DT:
            dt = now - last_send
            dt = max(0.01, min(dt, 0.05))
            last_send = now

            # ---------- X PID -> Roll ----------
            err_x = SETPOINT_X_MM - x_filt
            x_i += err_x * dt
            x_i = max(-I_LIMIT_X, min(I_LIMIT_X, x_i))
            dx = 0.0 if prev_x is None else (x_filt - prev_x) / dt
            prev_x = x_filt
            u_roll = (Kp_x * err_x) + (Ki_x * x_i) - (Kd_x * dx)
            u_roll = max(-MAX_ROLL_US, min(MAX_ROLL_US, u_roll))
            if X_REVERSE: u_roll = -u_roll
            roll_rc = clamp(RC_MID + u_roll)

            # ---------- Y PID -> Pitch ----------
            err_y = SETPOINT_Y_MM - y_filt
            y_i += err_y * dt
            y_i = max(-I_LIMIT_Y, min(I_LIMIT_Y, y_i))
            dy = 0.0 if prev_y is None else (y_filt - prev_y) / dt
            prev_y = y_filt
            u_pitch = (Kp_y * err_y) + (Ki_y * y_i) - (Kd_y * dy)
            u_pitch = max(-MAX_PITCH_US, min(MAX_PITCH_US, u_pitch))
            if Y_REVERSE: u_pitch = -u_pitch
            pitch_rc = clamp(RC_MID + u_pitch)

            # ---------- ANG PID -> Yaw ----------
            # use wrapped angle error so it always takes shortest direction
            err_a = wrap_deg180(SETPOINT_ANG_DEG - a_filt)
            a_i += err_a * dt
            a_i = max(-I_LIMIT_A, min(I_LIMIT_A, a_i))
            da = 0.0 if prev_a is None else wrap_deg180(a_filt - prev_a) / dt
            prev_a = a_filt
            u_yaw = (Kp_a * err_a) + (Ki_a * a_i) - (Kd_a * da)
            u_yaw = max(-MAX_YAW_US, min(MAX_YAW_US, u_yaw))
            if ANG_REVERSE: u_yaw = -u_yaw
            yaw_rc = clamp(RC_MID + u_yaw)

            # throttle stays low for testing
            throttle_rc = RC_MIN

            # send only when marker seen
            send_rc(seq, roll_rc, pitch_rc, throttle_rc, yaw_rc)
            seq = (seq + 1) & 0xFF

        # overlay
        cv2.putText(frame,
                    f"X:{x_filt:+.1f} R:{roll_rc}  Y:{y_filt:+.1f} P:{pitch_rc}  ANG:{a_filt:+.1f} Y:{yaw_rc}  Z:{z_mm:+.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 0),
                    2)

        marker_was_seen = True

    else:
        # marker lost -> stop sending UDP -> manual fallback
        if marker_was_seen:
            x_i = 0.0; prev_x = None; x_filt = None
            y_i = 0.0; prev_y = None; y_filt = None
            a_i = 0.0; prev_a = None; a_filt = None
            marker_was_seen = False

        now = time.time()
        if (now - last_print) >= PRINT_DT:
            last_print = now
            print("NO MARKER (not sending UDP -> manual fallback)")

        cv2.putText(frame, "NO MARKER", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Aruco X/Y/ANG PID -> Roll/Pitch/Yaw", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
