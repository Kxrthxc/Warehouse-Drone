import socket
import struct
import time

ESP_IP = "192.168.4.1"   # AP mode ESP32 default
# ESP_IP = "10.30.78.48" # STA mode (use Serial Monitor printed IP)
ESP_PORT = 14550

def clamp(v, lo=1000, hi=2000):
    return max(lo, min(hi, int(v)))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Start values (edit these to test)
roll     = 1800
pitch    = 1800
throttle = 1000
yaw      = 1500

rate_hz = 50
dt = 1.0 / rate_hz

print("Sending RC to", (ESP_IP, ESP_PORT))
print("Ctrl+C to stop.")

seq = 0
try:
    while True:
        roll     = clamp(roll)
        pitch    = clamp(pitch)
        throttle = clamp(throttle)
        yaw      = clamp(yaw)

        payload = b"RC" + struct.pack("<4H", roll, pitch, throttle, yaw) + struct.pack("<B", seq & 0xFF)
        csum = sum(payload) & 0xFF
        packet = payload + struct.pack("<B", csum)

        sock.sendto(packet, (ESP_IP, ESP_PORT))

        # print occasionally
        if seq % 25 == 0:
            print(f"R:{roll} P:{pitch} T:{throttle} Y:{yaw}")

        seq += 1
        time.sleep(dt)

except KeyboardInterrupt:
    print("\nStopped.")
