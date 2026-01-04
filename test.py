#!/usr/bin/env python3
"""
UDP RC Controller for ESP32 IBUS Bridge
Sends test RC channel data to ESP32 via UDP
"""

import socket
import time
import math
import sys

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
ESP32_IP = "10.30.78.148"  # <--- Your ESP32 IP address
UDP_PORT = 44444
SEND_RATE = 50  # Hz (packets per second)
# ==========================================

def clamp(value, min_val=1000, max_val=2000):
    """Ensure value is within valid RC range"""
    return max(min_val, min(max_val, int(value)))

def main():
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print("=" * 60)
    print("  ESP32 IBUS Bridge - UDP RC Controller")
    print("=" * 60)
    print(f"Target: {ESP32_IP}:{UDP_PORT}")
    print(f"Send Rate: {SEND_RATE} Hz")
    print("\nTest Modes:")
    print("  1. Static Center (all 1500) - 5 seconds")
    print("  2. Throttle Ramp (1000→1500) - 5 seconds")
    print("  3. Roll Oscillation - 5 seconds")
    print("  4. All Channels Test - Continuous")
    print("\nPress CTRL+C to stop anytime")
    print("=" * 60)
    print()
    
    interval = 1.0 / SEND_RATE
    packet_count = 0
    
    try:
        # Test 1: Static center position
        print("\n[Test 1/4] Static Center Position (5s)...")
        start = time.time()
        while time.time() - start < 5:
            roll = pitch = throttle = yaw = 1500
            packet = f"{roll},{pitch},{throttle},{yaw}"
            sock.sendto(packet.encode(), (ESP32_IP, UDP_PORT))
            packet_count += 1
            print(f"  Sent #{packet_count}: {packet}", end='\r')
            time.sleep(interval)
        
        # Test 2: Throttle ramp up
        print("\n[Test 2/4] Throttle Ramp (1000→1500, 5s)...")
        start = time.time()
        while time.time() - start < 5:
            elapsed = time.time() - start
            throttle = int(1000 + (elapsed / 5.0) * 500)  # Ramp from 1000 to 1500
            roll = pitch = yaw = 1500
            packet = f"{roll},{pitch},{throttle},{yaw}"
            sock.sendto(packet.encode(), (ESP32_IP, UDP_PORT))
            packet_count += 1
            print(f"  Sent #{packet_count}: {packet}", end='\r')
            time.sleep(interval)
        
        # Test 3: Roll oscillation
        print("\n[Test 3/4] Roll Oscillation (1200↔1800, 5s)...")
        start = time.time()
        while time.time() - start < 5:
            t = time.time()
            roll = int(1500 + 300 * math.sin(t * 2))  # Oscillate ±300 from center
            pitch = yaw = 1500
            throttle = 1200  # Low throttle
            packet = f"{roll},{pitch},{throttle},{yaw}"
            sock.sendto(packet.encode(), (ESP32_IP, UDP_PORT))
            packet_count += 1
            print(f"  Sent #{packet_count}: {packet}", end='\r')
            time.sleep(interval)
        
        # Test 4: Continuous all-channel test
        print("\n[Test 4/4] All Channels Test (Continuous)...")
        print("  Roll & Pitch: slow sine waves")
        print("  Throttle: 1100-1400 oscillation")
        print("  Yaw: centered at 1500")
        print()
        
        while True:
            t = time.time()
            
            # Different frequencies for each channel
            roll = int(1500 + 400 * math.sin(t * 0.5))      # Slow roll
            pitch = int(1500 + 400 * math.cos(t * 0.7))     # Slow pitch
            throttle = int(1250 + 150 * math.sin(t * 1.5))  # Medium throttle oscillation
            yaw = 1500  # Keep yaw centered
            
            # Clamp all values to valid range
            roll = clamp(roll)
            pitch = clamp(pitch)
            throttle = clamp(throttle)
            yaw = clamp(yaw)
            
            packet = f"{roll},{pitch},{throttle},{yaw}"
            sock.sendto(packet.encode(), (ESP32_IP, UDP_PORT))
            packet_count += 1
            
            print(f"  📡 #{packet_count:05d} | R:{roll:4d} P:{pitch:4d} T:{throttle:4d} Y:{yaw:4d}", end='\r')
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("🛑 Stopped by user")
        print(f"📊 Total packets sent: {packet_count}")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    finally:
        sock.close()

if __name__ == "__main__":
    main()