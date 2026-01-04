import cv2
import numpy as np
import time

# ================================
# 🔧 USER SETTINGS
# ================================
STREAM_URL = "http://10.30.78.48/stream"   # <--- put your ESP32 stream URL
PATTERN_SIZE = (6, 8)                           # inner corners (rows, cols)
SQUARE_SIZE = 25.0                               # mm
MAX_FRAMES = 20                                  # number of valid captures required
SAVE_FILE = "camera_calibration.npz"
# ================================


# Prepare object points (0,0,0), (1,0,0) ... multiplied by square size
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Storage for points
objpoints = []
imgpoints = []

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("❌ ERROR: Cannot open stream. Check URL or network!")
    exit()

print("\n📸 Calibration Started")
print("Press SPACE to capture a valid frame.")
print("Need at least", MAX_FRAMES, "frames.\n")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Warning: Failed to grab frame.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply advanced flags for noisy ESP32 feed
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)

    display = frame.copy()

    if found:
        cv2.drawChessboardCorners(display, PATTERN_SIZE, corners, found)
        text = "Chessboard DETECTED. Press SPACE to capture."
        color = (0, 255, 0)
    else:
        text = "Chessboard NOT detected. Adjust angle/distance..."
        color = (0, 0, 255)

    cv2.putText(display, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Calibration Feed", display)
    key = cv2.waitKey(1) & 0xFF

    # SPACE = Capture this frame if chessboard detected
    if key == 32 and found:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        imgpoints.append(corners2)
        objpoints.append(objp)
        count += 1
        print(f"✔ Captured frame {count}/{MAX_FRAMES}")

        # Pause to avoid double capture
        time.sleep(0.4)

        if count >= MAX_FRAMES:
            break

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ================================
# 🔧 RUN CALIBRATION
# ================================
print("\n📐 Running Calibration...")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\n========================")
print("📏 CALIBRATION RESULTS")
print("========================")
print("RMS error:", ret)
print("\nCamera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# Save result
np.savez(SAVE_FILE,
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs,
         rvecs=rvecs,
         tvecs=tvecs)

print("\n💾 Saved calibration to:", SAVE_FILE)
print("🎉 Calibration completed successfully!")
