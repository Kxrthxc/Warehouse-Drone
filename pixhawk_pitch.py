import cv2
import numpy as np
import socket
import struct
import time

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
# PITCH CONTROL (Y -> CH2)
# ================================
# When marker is visually centered, your Yraw isn't 0.
# Put that "center Yraw" here so the centered value becomes 0.
Y_CENTER_OFFSET_MM = -35.0     # <-- CHANGE THIS after reading your center value

SETPOINT_Y_MM = 0.0            # centered = 0 now

Kp_y = 0.7
Ki_y = 0
Kd_y = 0.12

MAX_PITCH_US = 40
I_LIMIT_Y    = 30
Y_LPF_ALPHA  = 0.25
Y_REVERSE    = True

# ================================
# PID STATE
# ================================
y_i = 0.0
prev_y = None
y_filt = None

# ================================
# UDP SOCKET
# ================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def clamp(v, lo=RC_MIN, hi=RC_MAX):
    return max(lo, min(hi, int(v)))

def send_pitch_only(seq, pitch_rc):
    roll_rc     = RC_MID
    throttle_rc = RC_MIN
    yaw_rc      = RC_MID

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

print("▶ PITCH ONLY OVERRIDE sender running.")
print("  - Uses y_centered = Yraw - Y_CENTER_OFFSET_MM")
print("  - SETPOINT_Y_MM = 0 means marker centered")
print("Press 'q' to exit.\n")

seq = 0
last_send = time.time()
last_print = time.time()
marker_was_seen = False
pitch_rc = RC_MID

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

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, camera_matrix, dist_coeffs
        )

        marker_id = int(ids[0][0])
        tvec = tvecs[0][0]
        x_mm, y_mm, z_mm = float(tvec[0]), float(tvec[1]), float(tvec[2])

        # ---- CENTERED Y ----
        y_centered = y_mm - Y_CENTER_OFFSET_MM

        # low-pass filter on centered value
        if y_filt is None:
            y_filt = y_centered
        else:
            y_filt = (Y_LPF_ALPHA * y_centered) + ((1.0 - Y_LPF_ALPHA) * y_filt)

        # PID update @ SEND_DT
        if (now - last_send) >= SEND_DT:
            dt = now - last_send
            dt = max(0.01, min(dt, 0.05))
            last_send = now

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

            send_pitch_only(seq, pitch_rc)
            seq = (seq + 1) & 0xFF

        # terminal print
        if (now - last_print) >= PRINT_DT:
            last_print = now
            print(f"ID:{marker_id}  Yraw:{y_mm:+.1f}  Yc:{y_centered:+.1f}  Yf:{y_filt:+.1f}  PITCH_RC:{pitch_rc}")

        cv2.putText(frame, f"Yc:{y_filt:+.1f}mm  PITCH_RC:{pitch_rc}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        marker_was_seen = True

    else:
        if marker_was_seen:
            y_i = 0.0
            prev_y = None
            y_filt = None
            marker_was_seen = False

        if (now - last_print) >= PRINT_DT:
            last_print = now
            print("NO MARKER (not sending UDP -> ESP should stay manual)")

        cv2.putText(frame, "NO MARKER (UDP STOP)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Pitch Override Only", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
