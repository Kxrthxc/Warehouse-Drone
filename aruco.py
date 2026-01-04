import cv2
import numpy as np
import math

# ================================
# 🔧 USER SETTINGS
# ================================
STREAM_URL = "http://10.35.170.148/stream"   # ESP32-CAM stream
MARKER_LENGTH = 50.0  # mm (side length of your ArUco marker)
CALIB_FILE = "camera_calibration.npz"
ARUCO_DICT = cv2.aruco.DICT_6X6_250
# ================================

# Load calibration
data = np.load(CALIB_FILE)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

print("Loaded calibration:")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coeffs:\n", dist_coeffs)

# Init ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Connect video stream
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("❌ Cannot open ESP32-CAM stream!")
    exit()

print("\n▶ ArUco Detection running...")
print("Press 'q' to exit.\n")

def rotationMatrixToEulerAngles(R):
    """
    Convert rotation matrix to Euler angles (yaw, pitch, roll)
    R: 3x3 rotation matrix
    Returns angles in degrees
    """
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

    # Convert from radians to degrees
    return np.degrees([x, y, z])

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Frame read error.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Aruco markers
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Pose estimation for each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            MARKER_LENGTH,
            camera_matrix,
            dist_coeffs
        )

        for i in range(len(ids)):
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]

            # Draw 3D axis on marker
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 40)

            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = rotationMatrixToEulerAngles(R)

            # Display translation vector + yaw
            x, y, z = tvec
            cv2.putText(frame,
                        f"ID:{ids[i][0]} X:{x:.1f} Y:{y:.1f} Z:{z:.1f} Yaw:{yaw:.1f}",
                        (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)
            print(f"ID:{ids[i][0]} X:{x:.1f} Y:{y:.1f} Z:{z:.1f} Yaw:{yaw:.1f}")
    cv2.imshow("Aruco Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
