import cv2
import numpy as np
import math
import socket
import struct
import time

# ================================
# 🔧 USER SETTINGS
# ================================
STREAM_URL = "http://10.30.78.48/stream"     # ESP32-CAM stream
ESP_IP     = "10.30.78.48"                   # ESP32 IP for UDP (STA mode)
# ESP_IP   = "192.168.4.1"                   # AP mode (if you use ESP AP)
ESP_PORT   = 14550

MARKER_LENGTH = 50.0                         # mm (side length)
CALIB_FILE = "camera_calibration.npz"
ARUCO_DICT = cv2.aruco.DICT_6X6_250

SEND_HZ = 50                                 # UDP send rate
SEND_DT = 1.0 / SEND_HZ

# ================================
# RC DEFAULTS (safe / neutral)
# ================================
RC_MIN, RC_MID, RC_MAX = 1000, 1500, 2000

roll_rc     = RC_MID
pitch_rc    = RC_MID
throttle_rc = RC_MIN   # keep low during bench tests
yaw_rc      = RC_MID

# ================================
# LOAD CALIBRATION
# ================================
data = np.load(CALIB_FILE)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

print("Loaded calibration:")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coeffs:\n", dist_coeffs)

# ================================
# INIT ARUCO
# ================================
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ================================
# UDP SOCKET
# ================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def clamp(v, lo=RC_MIN, hi=RC_MAX):
    return max(lo, min(hi, int(v)))

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.degrees([x, y, z])  # roll, pitch, yaw

def send_rc(seq, roll, pitch, throttle, yaw):
    # Packet: 'R''C' + 4xU16 + seq(u8) + checksum(u8) => 12 bytes
    payload = b"RC" + struct.pack("<4H", roll, pitch, throttle, yaw) + struct.pack("<B", seq & 0xFF)
    csum = sum(payload) & 0xFF
    packet = payload + struct.pack("<B", csum)
    sock.sendto(packet, (ESP_IP, ESP_PORT))

# ================================
# CONNECT STREAM
# ================================
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("❌ Cannot open ESP32-CAM stream:", STREAM_URL)
    raise SystemExit

print("\n▶ ArUco Detection + RC UDP sender running...")
print("Stream:", STREAM_URL)
print("UDP ->", (ESP_IP, ESP_PORT))
print("✅ IMPORTANT: When marker is NOT detected, this script STOPS sending UDP.")
print("   Then ESP32 will automatically fall back to MANUAL radio control.\n")
print("Press 'q' to exit.\n")

seq = 0
last_send = 0.0
last_print = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    marker_seen = (ids is not None and len(ids) > 0)

    if marker_seen:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            MARKER_LENGTH,
            camera_matrix,
            dist_coeffs
        )

        # Use first marker only
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]
        x, y, z = tvec

        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 40)

        R, _ = cv2.Rodrigues(rvec)
        roll_deg, pitch_deg, yaw_deg = rotationMatrixToEulerAngles(R)

        # ====== YOUR CONTROL LOGIC HERE ======
        # For now, just send neutral sticks (edit later)
        roll_rc     = RC_MIN
        pitch_rc    = RC_MID
        yaw_rc      = RC_MID
        throttle_rc = RC_MIN   # keep low for safety while testing

        cv2.putText(frame,
                    f"ID:{ids[0][0]} X:{x:.1f} Y:{y:.1f} Z:{z:.1f} Yaw:{yaw_deg:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    # ================================
    # SEND UDP ONLY WHEN MARKER IS SEEN
    # ================================
    now = time.time()
    if marker_seen and (now - last_send >= SEND_DT):
        send_rc(seq,
                clamp(roll_rc),
                clamp(pitch_rc),
                clamp(throttle_rc),
                clamp(yaw_rc))
        seq += 1
        last_send = now

    # occasional console print
    if now - last_print >= 0.5:
        last_print = now
        if marker_seen:
            print(f"[SEND] R:{roll_rc} P:{pitch_rc} T:{throttle_rc} Y:{yaw_rc}")
        else:
            print("[NO MARKER] Not sending UDP -> ESP32 falls back to MANUAL")

    cv2.imshow("Aruco Tracking + RC", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
