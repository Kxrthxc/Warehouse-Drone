import cv2
import numpy as np
import socket
import struct
import time
import math

# ================================
# SETTINGS
# ================================
STREAM_URL = "http://10.78.238.148/stream"
ESP_IP     = "10.78.238.148"
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
# CENTER OFFSETS (center marker => 0)
# ================================
Y_CENTER_OFFSET_MM = -35.0
X_CENTER_OFFSET_MM = +20.0

# YAW: marker "straight" angle may not be 0deg; put that here after reading ANGraw at straight
ANG_CENTER_OFFSET_DEG = 0.0     # <-- tune later if needed

# ================================
# SETPOINTS (after centering)
# ================================
SETPOINT_Y_MM    = 0.0
SETPOINT_X_MM    = 0.0
SETPOINT_ANG_DEG = 0.0          # yaw target

# ================================
# PITCH PID (Y -> CH2)
# ================================
Kp_y = 0.7
Ki_y = 0.0
Kd_y = 0.12

MAX_PITCH_US = 40
I_LIMIT_Y    = 30
Y_LPF_ALPHA  = 0.25
Y_REVERSE    = True

# ================================
# ROLL PID (X -> CH1)
# ================================
Kp_x = 0.7
Ki_x = 0.0
Kd_x = 0.12

MAX_ROLL_US = 40
I_LIMIT_X   = 30
X_LPF_ALPHA = 0.25
X_REVERSE   = True

# ================================
# YAW PID (ANG -> CH4)
# ================================
Kp_a = 0.7
Ki_a = 0.0
Kd_a = 0.12

MAX_YAW_US = 50
I_LIMIT_A  = 30
A_LPF_ALPHA = 0.25
A_REVERSE   = True

# ================================
# PID STATE
# ================================
y_i = 0.0
prev_y = None
y_filt = None

x_i = 0.0
prev_x = None
x_filt = None

a_i = 0.0
prev_a = None
a_filt = None

# ================================
# UDP SOCKET
# ================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def clamp(v, lo=RC_MIN, hi=RC_MAX):
    return max(lo, min(hi, int(v)))

def wrap_deg180(a):
    """Wrap any angle to [-180, +180)."""
    return (a + 180.0) % 360.0 - 180.0

def send_rpy(seq, roll_rc, pitch_rc, yaw_rc):
    """
    Packet format:
      'RC' + roll + pitch + throttle + yaw + seq + checksum
    """
    throttle_rc = RC_MIN
    payload = b"RC" + struct.pack("<4H", roll_rc, pitch_rc, throttle_rc, yaw_rc) + struct.pack("<B", seq & 0xFF)
    csum = sum(payload) & 0xFF
    packet = payload + struct.pack("<B", csum)
    sock.sendto(packet, (ESP_IP, ESP_PORT))

# ================================
# LOAD CALIBRATION
# ================================
data = np.load(CALIB_FILE)
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

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

print("▶ ROLL + PITCH + YAW sender running.")
print("  - CH1=ROLL from X, CH2=PITCH from Y, CH4=YAW from marker angle")
print("Press 'q' to exit.\n")

seq = 0
last_send = time.time()
last_print = time.time()
marker_was_seen = False

pitch_rc = RC_MID
roll_rc  = RC_MID
yaw_rc   = RC_MID

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    marker_seen = (ids is not None and len(ids) > 0)

    now = time.time()

    if marker_seen:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, camera_matrix, dist_coeffs
        )

        marker_id = int(ids[0][0])
        tvec = tvecs[0][0]
        x_mm, y_mm, z_mm = float(tvec[0]), float(tvec[1]), float(tvec[2])

        # ---- CENTERED X/Y ----
        x_centered = x_mm - X_CENTER_OFFSET_MM
        y_centered = y_mm - Y_CENTER_OFFSET_MM

        # ---- YAW ANGLE from marker corners (image 2D angle) ----
        pts = corners[0][0]  # (4,2) order
        (px0, py0) = pts[0]
        (px1, py1) = pts[1]
        ang_raw = math.degrees(math.atan2((py1 - py0), (px1 - px0)))
        ang_raw = wrap_deg180(ang_raw)
        ang_centered = wrap_deg180(ang_raw - ANG_CENTER_OFFSET_DEG)

        # ---- LPF ----
        if x_filt is None:
            x_filt = x_centered
        else:
            x_filt = (X_LPF_ALPHA * x_centered) + ((1.0 - X_LPF_ALPHA) * x_filt)

        if y_filt is None:
            y_filt = y_centered
        else:
            y_filt = (Y_LPF_ALPHA * y_centered) + ((1.0 - Y_LPF_ALPHA) * y_filt)

        if a_filt is None:
            a_filt = ang_centered
        else:
            a_filt = wrap_deg180((A_LPF_ALPHA * ang_centered) + ((1.0 - A_LPF_ALPHA) * a_filt))

        # ---- PID update @ SEND_DT ----
        if (now - last_send) >= SEND_DT:
            dt = now - last_send
            dt = max(0.01, min(dt, 0.05))
            last_send = now

            # ===== ROLL PID (X -> CH1) =====
            err_x = SETPOINT_X_MM - x_filt
            x_i += err_x * dt
            x_i = max(-I_LIMIT_X, min(I_LIMIT_X, x_i))
            dx = 0.0 if prev_x is None else (x_filt - prev_x) / dt
            prev_x = x_filt

            u_roll = (Kp_x * err_x) + (Ki_x * x_i) - (Kd_x * dx)
            u_roll = max(-MAX_ROLL_US, min(MAX_ROLL_US, u_roll))
            if X_REVERSE:
                u_roll = -u_roll
            roll_rc = clamp(RC_MID + u_roll)

            # ===== PITCH PID (Y -> CH2) =====
            err_y = SETPOINT_Y_MM - y_filt
            y_i += err_y * dt
            y_i = max(-I_LIMIT_Y, min(I_LIMIT_Y, y_i))
            dy = 0.0 if prev_y is None else (y_filt - prev_y) / dt
            prev_y = y_filt

            u_pitch = (Kp_y * err_y) + (Ki_y * y_i) - (Kd_y * dy)
            u_pitch = max(-MAX_PITCH_US, min(MAX_PITCH_US, u_pitch))
            if Y_REVERSE:
                u_pitch = -u_pitch
            pitch_rc = clamp(RC_MID + u_pitch)

            # ===== YAW PID (ANG -> CH4) =====
            err_a = wrap_deg180(SETPOINT_ANG_DEG - a_filt)  # shortest way
            a_i += err_a * dt
            a_i = max(-I_LIMIT_A, min(I_LIMIT_A, a_i))
            da = 0.0 if prev_a is None else wrap_deg180(a_filt - prev_a) / dt
            prev_a = a_filt

            u_yaw = (Kp_a * err_a) + (Ki_a * a_i) - (Kd_a * da)
            u_yaw = max(-MAX_YAW_US, min(MAX_YAW_US, u_yaw))
            if A_REVERSE:
                u_yaw = -u_yaw
            yaw_rc = clamp(RC_MID + u_yaw)

            # SEND
            send_rpy(seq, roll_rc, pitch_rc, yaw_rc)
            seq = (seq + 1) & 0xFF

        # ---- terminal print ----
        if (now - last_print) >= PRINT_DT:
            last_print = now
            print(
                f"ID:{marker_id}  "
                f"Xraw:{x_mm:+.1f} Xc:{x_centered:+.1f} Xf:{x_filt:+.1f}  R:{roll_rc}  |  "
                f"Yraw:{y_mm:+.1f} Yc:{y_centered:+.1f} Yf:{y_filt:+.1f}  P:{pitch_rc}  |  "
                f"ANGraw:{ang_raw:+.1f} Ac:{ang_centered:+.1f} Af:{a_filt:+.1f}  Y:{yaw_rc}"
            )

        cv2.putText(frame,
                    f"Xf:{x_filt:+.1f} R:{roll_rc}  Yf:{y_filt:+.1f} P:{pitch_rc}  Af:{a_filt:+.1f} Y:{yaw_rc}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)

        marker_was_seen = True

    else:
        # marker lost -> stop sending UDP -> ESP manual fallback
        if marker_was_seen:
            x_i = 0.0; prev_x = None; x_filt = None
            y_i = 0.0; prev_y = None; y_filt = None
            a_i = 0.0; prev_a = None; a_filt = None
            marker_was_seen = False

        if (now - last_print) >= PRINT_DT:
            last_print = now
            print("NO MARKER (UDP STOP -> manual fallback)")

        cv2.putText(frame, "NO MARKER (UDP STOP)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Roll + Pitch + Yaw Override", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
