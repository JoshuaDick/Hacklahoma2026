#!/usr/bin/env python3
#100.89.208.16
#!/usr/bin/env python3
"""
Spatial Streamer Receiver
Receives camera feed and spatial data from Android device via Tailscale
"""

import socket
import struct
import json
import cv2
import numpy as np
from datetime import datetime
import threading
import time

class SpatialStreamReceiver:
    def __init__(self, host='0.0.0.0', port=8888):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        
        # Display windows
        self.video_window = "Camera Feed"
        self.spatial_window = "Spatial Data"
        
        # Latest spatial data
        self.latest_spatial_data = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
    def start_server(self):
        """Start the TCP server to listen for incoming connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        
        print(f"Server listening on {self.host}:{self.port}")
        print("Waiting for Android device to connect...")
        print("Make sure both devices are connected via Tailscale")
        
        self.client_socket, client_address = self.server_socket.accept()
        print(f"Connected to {client_address}")
        
        self.running = True
        return True
    
    def receive_exact(self, num_bytes):
        """Receive exact number of bytes from socket"""
        data = b''
        while len(data) < num_bytes:
            chunk = self.client_socket.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data
    
    def receive_frame(self):
        """Receive one frame with spatial data"""
        try:
            # Read JSON length
            json_length_bytes = self.receive_exact(4)
            json_length = struct.unpack('!I', json_length_bytes)[0]
            
            # Read JSON data
            json_bytes = self.receive_exact(json_length)
            spatial_data = json.loads(json_bytes.decode('utf-8'))
            
            # Read image length
            image_length_bytes = self.receive_exact(4)
            image_length = struct.unpack('!I', image_length_bytes)[0]
            
            # Read image data
            image_data = self.receive_exact(image_length)
            
            return spatial_data, image_data
            
        except Exception as e:
            print(f"Error receiving frame: {e}")
            return None, None
    
    def create_spatial_visualization(self, width=800, height=600):
        """Create a visualization of spatial data"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if self.latest_spatial_data is None:
            cv2.putText(img, "Waiting for data...", (50, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return img
        
        # Extract data
        accel = self.latest_spatial_data['accelerometer']
        gyro = self.latest_spatial_data['gyroscope']
        mag = self.latest_spatial_data['magnetometer']
        orient = self.latest_spatial_data['orientation']
        
        # Display info
        y_offset = 40
        line_height = 35
        
        # Title
        cv2.putText(img, "Spatial Data Visualization", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += line_height + 10
        
        # FPS
        cv2.putText(img, f"FPS: {self.fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # Accelerometer
        cv2.putText(img, "Accelerometer (m/s²):", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        y_offset += line_height
        cv2.putText(img, f"  X: {accel[0]:7.3f}  Y: {accel[1]:7.3f}  Z: {accel[2]:7.3f}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Draw accelerometer bars
        self.draw_3d_bars(img, accel, 20, y_offset, "Accel", scale=20)
        y_offset += 100
        
        # Gyroscope
        cv2.putText(img, "Gyroscope (rad/s):", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        y_offset += line_height
        cv2.putText(img, f"  X: {gyro[0]:7.3f}  Y: {gyro[1]:7.3f}  Z: {gyro[2]:7.3f}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Draw gyroscope bars
        self.draw_3d_bars(img, gyro, 20, y_offset, "Gyro", scale=50)
        y_offset += 100
        
        # Orientation (Azimuth, Pitch, Roll)
        cv2.putText(img, "Orientation (radians):", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        y_offset += line_height
        cv2.putText(img, f"  Azimuth: {orient[0]:7.3f}  Pitch: {orient[1]:7.3f}  Roll: {orient[2]:7.3f}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Draw orientation visualization
        self.draw_orientation_3d(img, orient, width - 250, 150, 200)
        
        # Magnetometer
        y_offset += 20
        cv2.putText(img, "Magnetometer (μT):", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        y_offset += line_height
        cv2.putText(img, f"  X: {mag[0]:7.3f}  Y: {mag[1]:7.3f}  Z: {mag[2]:7.3f}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def draw_3d_bars(self, img, values, x, y, label, scale=50):
        """Draw 3D bar chart for sensor values"""
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue for X, Y, Z
        labels = ['X', 'Y', 'Z']
        bar_width = 60
        spacing = 80
        
        for i, (value, color, lab) in enumerate(zip(values, colors, labels)):
            bar_x = x + i * spacing
            bar_height = int(value * scale)
            center_y = y + 50
            
            if bar_height > 0:
                cv2.rectangle(img, (bar_x, center_y - bar_height), 
                            (bar_x + bar_width, center_y), color, -1)
            else:
                cv2.rectangle(img, (bar_x, center_y), 
                            (bar_x + bar_width, center_y - bar_height), color, -1)
            
            # Baseline
            cv2.line(img, (bar_x, center_y), (bar_x + bar_width, center_y), (128, 128, 128), 2)
            
            # Label
            cv2.putText(img, lab, (bar_x + 20, center_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_orientation_3d(self, img, orient, cx, cy, size):
        """Draw 3D orientation visualization"""
        azimuth, pitch, roll = orient
        
        # Draw circle background
        cv2.circle(img, (cx, cy), size, (50, 50, 50), -1)
        cv2.circle(img, (cx, cy), size, (100, 100, 100), 2)
        
        # Draw axes
        # X-axis (red) - affected by roll
        x_end_x = int(cx + size * 0.8 * np.cos(roll))
        x_end_y = int(cy + size * 0.8 * np.sin(roll))
        cv2.line(img, (cx, cy), (x_end_x, x_end_y), (0, 0, 255), 3)
        
        # Y-axis (green) - affected by pitch
        y_end_x = int(cx + size * 0.8 * np.cos(pitch + np.pi/2))
        y_end_y = int(cy + size * 0.8 * np.sin(pitch + np.pi/2))
        cv2.line(img, (cx, cy), (y_end_x, y_end_y), (0, 255, 0), 3)
        
        # Z-axis indication (blue) - affected by azimuth
        z_end_x = int(cx + size * 0.6 * np.cos(azimuth))
        z_end_y = int(cy + size * 0.6 * np.sin(azimuth))
        cv2.line(img, (cx, cy), (z_end_x, z_end_y), (255, 0, 0), 3)
        
        # Center dot
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)
        
        # Labels
        cv2.putText(img, "Orientation", (cx - 60, cy - size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def run(self):
        """Main receiver loop"""
        if not self.start_server():
            return
        
        cv2.namedWindow(self.video_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.spatial_window, cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                spatial_data, image_data = self.receive_frame()
                
                if spatial_data is None or image_data is None:
                    print("Connection lost")
                    break
                
                self.latest_spatial_data = spatial_data
                self.update_fps()
                
                # Decode image (assuming YUV format from Android)
                try:
                    # Try to decode as JPEG or create from raw data
                    nparr = np.frombuffer(image_data, np.uint8)
                    
                    # Try JPEG decode first
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        # If not JPEG, try to reshape as grayscale
                        width = spatial_data['imageWidth']
                        height = spatial_data['imageHeight']
                        
                        # This is a simplified approach
                        if len(image_data) >= width * height:
                            frame = nparr[:width*height].reshape((height, width))
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    if frame is not None:
                        # Add overlay info
                        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Size: {frame.shape[1]}x{frame.shape[0]}", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow(self.video_window, frame)
                    
                except Exception as e:
                    print(f"Error decoding image: {e}")
                
                # Display spatial data visualization
                spatial_viz = self.create_spatial_visualization()
                cv2.imshow(self.spatial_window, spatial_viz)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        cv2.destroyAllWindows()
        print("Cleanup complete")

def main():
    print("="*60)
    print("Spatial Streamer Receiver")
    print("="*60)
    print("\nInstructions:")
    print("1. Ensure both devices are connected to Tailscale")
    print("2. Find this computer's Tailscale IP: tailscale ip")
    print("3. Enter that IP in the Android app")
    print("4. Press Connect on the Android device")
    print("5. Press 'q' to quit")
    print("\n" + "="*60 + "\n")
    
    receiver = SpatialStreamReceiver()
    receiver.run()

if __name__ == "__main__":
    main()