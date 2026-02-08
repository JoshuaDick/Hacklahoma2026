#!/usr/bin/env python3
"""
Ultra-Lightweight Visual Odometry Receiver
Uses only OpenCV's sparse optical flow - NO heavy SLAM library needed
Expected: 40-60 FPS on CPU for tracking
"""

import socket
import struct
import cv2
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from queue import Queue, Empty
from collections import deque
import pygame
import speech_recognition as sr
import requests
import json
from gtts import gTTS
import pygame
from mutagen.mp3 import MP3
from pydub import AudioSegment
import sys
import sounddevice as sd
import numpy as np
from scipy.signal import find_peaks
import time
import pyaudio
import numpy as np

import pygame
import time

controller_state = {
    "lx": 0.0,
    "ly": 0.0,
    "rx": 0.0,
    "ry": 0.0,
    "buttons": {}
}

state_lock = threading.Lock()

#NOTE TO SELF: 53% microphone is ideal on laptop for clap detection

# Constants
THRESHOLD = 25000  
CLAP_GAP = 1.0  # Max time gap between claps (seconds)
CHUNK = 1024  # Buffer size
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate

def controller_input_thread(stop_event, controller_state, state_lock):
    import pygame
    import time

    pygame.init()
    pygame.joystick.init()

    joystick = None
    last_activity = time.time()
    DISCONNECT_TIMEOUT = 10.0  # seconds with no input to assume disconnect

    def clear_state():
        with state_lock:
            controller_state["lx"] = 0.0
            controller_state["ly"] = 0.0
            controller_state["rx"] = 0.0
            controller_state["ry"] = 0.0
            controller_state["buttons"] = {}

    def has_activity(lx, ly, rx, ry, buttons):
        if abs(lx) > 0.01 or abs(ly) > 0.01:
            return True
        if abs(rx) > 0.01 or abs(ry) > 0.01:
            return True
        return any(buttons.values())

    print("[Controller] Thread started")

    while not stop_event.is_set():
        try:
            # -----------------------------------
            # ENSURE CONTROLLER CONNECTED
            # -----------------------------------
            if joystick is None:
                pygame.joystick.quit()
                pygame.joystick.init()

                if pygame.joystick.get_count() == 0:
                    clear_state()
                    time.sleep(1.0)
                    continue

                joystick = pygame.joystick.Joystick(0)
                joystick.init()
                last_activity = time.time()

                print(f"[Controller] Connected: {joystick.get_name()}")

            pygame.event.pump()

            # -----------------------------------
            # READ INPUT
            # -----------------------------------
            lx = joystick.get_axis(0)
            ly = joystick.get_axis(1)
            rx = joystick.get_axis(3)
            ry = joystick.get_axis(4)

            buttons = {
                i: joystick.get_button(i)
                for i in range(joystick.get_numbuttons())
            }

            if has_activity(lx, ly, rx, ry, buttons):
                last_activity = time.time()

            # -----------------------------------
            # DISCONNECT DETECTION
            # -----------------------------------
            if time.time() - last_activity > DISCONNECT_TIMEOUT:
                raise RuntimeError("Controller inactive: assumed disconnected")

            with state_lock:
                controller_state["lx"] = lx
                controller_state["ly"] = ly
                controller_state["rx"] = rx
                controller_state["ry"] = ry
                controller_state["buttons"] = buttons

            time.sleep(0.01)

        except RuntimeError as e:
            print(f"[Controller] {e}")
            clear_state()

            if joystick is not None:
                try:
                    joystick.quit()
                except:
                    pass

            joystick = None
            time.sleep(1.0)

    clear_state()
    pygame.joystick.quit()
    pygame.quit()
    print("[Controller] Thread stopped")




def update_joystick_display(canvas, left_dot, right_dot):
    with state_lock:
        lx = controller_state["lx"]
        ly = controller_state["ly"]
        rx = controller_state["rx"]
        ry = controller_state["ry"]

    # Map -1..1 → canvas coords
    def map_axis(x, y, cx, cy, r):
        return (
            cx + x * r,
            cy + y * r
        )

    lx_pos = map_axis(lx, ly, 100, 100, 60)
    rx_pos = map_axis(rx, ry, 300, 100, 60)

    canvas.coords(left_dot,
                  lx_pos[0]-5, lx_pos[1]-5,
                  lx_pos[0]+5, lx_pos[1]+5)

    canvas.coords(right_dot,
                  rx_pos[0]-5, rx_pos[1]-5,
                  rx_pos[0]+5, rx_pos[1]+5)

    canvas.after(16, update_joystick_display, canvas, left_dot, right_dot)

def start_display(roo):
    root = roo
    root.title("Controller Input Display")

    canvas = tk.Canvas(root, width=400, height=200, bg="black")
    canvas.pack()

    # Left joystick
    canvas.create_oval(40, 40, 160, 160, outline="white")
    left_dot = canvas.create_oval(95, 95, 105, 105, fill="red")

    # Right joystick
    canvas.create_oval(240, 40, 360, 160, outline="white")
    right_dot = canvas.create_oval(295, 95, 305, 105, fill="blue")

    update_joystick_display(canvas, left_dot, right_dot)
    root.mainloop()


def listen_for_claps():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for claps...")
    clap_times = []

    try:
        while True:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            #print(np.max(data))
            if np.max(data) > THRESHOLD:
                now = time.time()
                print("Clap detected!")
                if clap_times and (now - clap_times[-1] <= CLAP_GAP):
                    print("Double clap detected!")
                    break
                clap_times = [now]
    except KeyboardInterrupt:
        print("\nStopped listening.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def get_mp3_length_in_ms(mp3_file_path):
    # Load the MP3 file
    audio = MP3(mp3_file_path)
    
    # Get the duration in seconds and convert it to milliseconds
    duration_in_seconds = audio.info.length
    duration_in_ms = duration_in_seconds * 1000
    
    return int(duration_in_ms)

def speak(str):
    tts = gTTS(text=str, lang='en', tld='com.au')
    tts.save("speech.mp3")
    time = get_mp3_length_in_ms("speech.mp3")
    slowSpeech = AudioSegment.from_mp3("speech.mp3")
    fastSpeech = slowSpeech.speedup(playback_speed=1.2)
    fastSpeech.export("speech.mp3",format='mp3')
    pygame.mixer.music.load("speech.mp3")
    pygame.mixer.music.set_volume(1.0)
    pygame.mixer.music.play(-1, 0.0)
    pygame.time.wait(int(time/1.2))
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()


class LightweightVisualOdometry:
    """
    Minimal visual odometry using Lucas-Kanade optical flow
    Much faster than full SLAM, good enough for trajectory tracking
    """
    def __init__(self, focal_length=500.0, pp=(320, 240)):
        # Camera intrinsics
        self.focal = focal_length
        self.pp = pp
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=300,  # More features for better stability
            qualityLevel=0.01,
            minDistance=15,  # Spread features out more
            blockSize=7
        )
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # State
        self.prev_gray = None
        self.prev_points = None
        self.trajectory = []
        self.current_pose = np.array([0.0, 0.0, 0.0])  # Just XYZ position
        
        # Scale and motion damping
        self.scale = 1.0  # Adaptive scale factor
        self.motion_damping = 0.3  # Reduce motion magnitude to prevent drift
        self.max_translation = 5.0  # Maximum translation per frame
        
        # Feature trail visualization
        self.trails = []
        self.max_trail_length = 20
        
        # Frame counter for drift correction
        self.frame_count = 0

    def detect_features(self, gray):
        """Detect good features to track"""
        return cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
    
    def track_features(self, prev_gray, curr_gray, prev_points):
        """Track features using Lucas-Kanade optical flow"""
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **self.lk_params
        )
        
        # Select good points
        if curr_points is not None and status is not None:
            # Flatten status array to 1D for proper indexing
            status = status.flatten()
            good_new = curr_points[status == 1]
            good_old = prev_points[status == 1]
            return good_new, good_old
        
        return None, None
    
    def estimate_motion(self, old_points, new_points):
        """Estimate camera motion from point correspondences"""
        if old_points is None or new_points is None or len(old_points) < 8:
            return None
        
        # Use homography-based motion estimation for more stability
        # This works better for planar or distant scenes
        try:
            # Try Essential matrix first (proper 3D motion)
            E, mask = cv2.findEssentialMat(
                old_points, new_points,
                focal=self.focal,
                pp=self.pp,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            
            if E is None or mask is None:
                return None
            
            # Count inliers
            inlier_count = np.sum(mask)
            if inlier_count < 8:
                return None
            
            # Recover pose
            _, R, t, pose_mask = cv2.recoverPose(E, old_points, new_points, 
                                                  focal=self.focal, pp=self.pp, mask=mask)
            
            # Calculate median optical flow magnitude for scale estimation
            flow = new_points - old_points
            flow_magnitude = np.linalg.norm(flow, axis=1)
            median_flow = np.median(flow_magnitude)
            
            # Adaptive scale based on flow magnitude
            # Less flow = less motion = smaller scale
            if median_flow > 0.5:  # Only update if significant motion
                self.scale = np.clip(median_flow / 10.0, 0.1, 2.0)
            
            # Apply scale and damping to translation
            t_scaled = t * self.scale * self.motion_damping
            
            # Clamp translation to prevent huge jumps
            t_magnitude = np.linalg.norm(t_scaled)
            if t_magnitude > self.max_translation:
                t_scaled = t_scaled / t_magnitude * self.max_translation
            
            return R, t_scaled, inlier_count / len(old_points)
            
        except Exception as e:
            return None
    
    def process_frame(self, frame):
        """
        Process a single frame and update trajectory
        Returns: (processed_frame, tracking_quality, num_features)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame - just detect features
        if self.prev_gray is None:
            self.prev_points = self.detect_features(gray)
            self.prev_gray = gray.copy()
            
            # Draw initial features
            if self.prev_points is not None:
                for point in self.prev_points:
                    x, y = point.ravel()
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            return frame, 100.0, len(self.prev_points) if self.prev_points is not None else 0
        
        # Track features
        curr_points, prev_points_good = self.track_features(self.prev_gray, gray, self.prev_points)
        
        if curr_points is None or len(curr_points) < 8:
            # Lost tracking - reinitialize
            self.prev_points = self.detect_features(gray)
            self.prev_gray = gray.copy()
            return frame, 0.0, 0
        
        # Estimate motion
        motion = self.estimate_motion(prev_points_good, curr_points)
        
        quality = 100.0 * len(curr_points) / max(1, len(self.prev_points))
        
        if motion is not None:
            R, t, inlier_ratio = motion
            
            # Simple 2D motion model: extract X and Z (ignore Y/height for now)
            # This gives us forward/backward and left/right motion
            dx = -t[0, 0]  # Left/right (negated for correct direction)
            dz = -t[2, 0]  # Forward/backward (negated for correct direction)
            
            # Update position (just accumulate 2D motion)
            self.current_pose[0] += dx
            self.current_pose[2] += dz
            
            # Store trajectory point
            self.trajectory.append(self.current_pose.copy())
            
            # Keep last 2000 points
            if len(self.trajectory) > 2000:
                self.trajectory.pop(0)
            
            # Display motion info
            motion_magnitude = np.sqrt(dx**2 + dz**2)
            cv2.putText(frame, f"Motion: {motion_magnitude:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Inliers: {inlier_ratio*100:.1f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        self.frame_count += 1
        
        # Update trails for visualization
        self.update_trails(prev_points_good, curr_points)
        
        # Draw tracking visualization
        frame = self.draw_tracking(frame, curr_points, prev_points_good)
        
        # Need more features? Add them
        if len(curr_points) < 100:
            # Create mask to avoid existing points
            mask = np.zeros_like(gray)
            mask[:] = 255
            for pt in curr_points:
                x, y = pt.ravel()
                cv2.circle(mask, (int(x), int(y)), 25, 0, -1)
            
            # Detect new features
            new_points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            
            if new_points is not None:
                curr_points = np.vstack([curr_points, new_points])
        
        # Update state
        self.prev_gray = gray.copy()
        self.prev_points = curr_points
        
        return frame, quality, len(curr_points)
    
    def update_trails(self, old_points, new_points):
        """Update feature trails for visualization"""
        if old_points is None or new_points is None:
            return
        
        for old, new in zip(old_points, new_points):
            a, b = new.ravel()
            c, d = old.ravel()
            
            # Add to trails
            found = False
            for trail in self.trails:
                if len(trail) > 0:
                    last_x, last_y = trail[-1]
                    if abs(last_x - c) < 5 and abs(last_y - d) < 5:
                        trail.append((int(a), int(b)))
                        if len(trail) > self.max_trail_length:
                            trail.pop(0)
                        found = True
                        break
            
            if not found and len(self.trails) < 200:
                self.trails.append([(int(a), int(b))])
        
        # Clean up old trails
        self.trails = [t for t in self.trails if len(t) > 0]
    
    def draw_tracking(self, frame, curr_points, prev_points):
        """Draw tracking visualization"""
        # Draw trails
        for trail in self.trails:
            if len(trail) > 1:
                for i in range(len(trail) - 1):
                    pt1 = trail[i]
                    pt2 = trail[i + 1]
                    # Fade color based on age
                    alpha = int(255 * (i + 1) / len(trail))
                    cv2.line(frame, pt1, pt2, (0, alpha, 255), 2)
        
        # Draw current points
        if curr_points is not None:
            for point in curr_points:
                x, y = point.ravel()
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Draw motion vectors
        if prev_points is not None and curr_points is not None:
            for old, new in zip(prev_points, curr_points):
                a, b = new.ravel()
                c, d = old.ravel()
                # Only draw if motion is significant
                if abs(a - c) > 1 or abs(b - d) > 1:
                    cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), 
                                   (255, 0, 0), 1, tipLength=0.3)
        
        return frame
    
    def draw_trajectory_map(self, width=400, height=400, scale=None):
        """Draw trajectory on a separate map view with auto-scaling"""
        map_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(self.trajectory) < 2:
            # No trajectory yet
            cv2.putText(map_img, "Building trajectory...", (50, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            return map_img
        
        # Convert trajectory to numpy array for easier processing
        traj_array = np.array(self.trajectory)
        
        # Auto-scale to fit in view
        if scale is None:
            # Calculate bounds
            x_min, x_max = traj_array[:, 0].min(), traj_array[:, 0].max()
            z_min, z_max = traj_array[:, 2].min(), traj_array[:, 2].max()
            
            # Add padding
            x_range = max(x_max - x_min, 1.0)
            z_range = max(z_max - z_min, 1.0)
            
            # Calculate scale to fit 80% of the view
            scale_x = (width * 0.8) / x_range
            scale_z = (height * 0.8) / z_range
            scale = min(scale_x, scale_z)
            
            # Calculate center offset
            center_x = (x_min + x_max) / 2
            center_z = (z_min + z_max) / 2
        else:
            center_x = 0
            center_z = 0
        
        # Convert 3D trajectory to 2D (bird's eye view - X,Z plane)
        points_2d = []
        for pos in self.trajectory:
            x = int(width / 2 + (pos[0] - center_x) * scale)
            z = int(height / 2 - (pos[2] - center_z) * scale)  # Negative because image Y is down
            points_2d.append((x, z))
        
        # Draw trajectory
        for i in range(len(points_2d) - 1):
            # Color gradient from blue (old) to green (new)
            ratio = i / max(1, len(points_2d) - 1)
            color = (
                int(255 * (1 - ratio)),  # Blue component
                int(255 * ratio),         # Green component
                0                         # Red component
            )
            
            # Only draw if points are within bounds
            if (0 <= points_2d[i][0] < width and 0 <= points_2d[i][1] < height and
                0 <= points_2d[i+1][0] < width and 0 <= points_2d[i+1][1] < height):
                cv2.line(map_img, points_2d[i], points_2d[i + 1], color, 2)
        
        # Draw current position
        if len(points_2d) > 0:
            curr_pt = points_2d[-1]
            if 0 <= curr_pt[0] < width and 0 <= curr_pt[1] < height:
                cv2.circle(map_img, curr_pt, 6, (0, 255, 0), -1)
                cv2.circle(map_img, curr_pt, 8, (255, 255, 255), 2)
        
        # Draw start position
        if len(points_2d) > 0:
            start_pt = points_2d[0]
            if 0 <= start_pt[0] < width and 0 <= start_pt[1] < height:
                cv2.circle(map_img, start_pt, 5, (255, 0, 0), -1)
        
        # Add grid
        grid_spacing = 50
        for i in range(0, width, grid_spacing):
            cv2.line(map_img, (i, 0), (i, height), (30, 30, 30), 1)
        for i in range(0, height, grid_spacing):
            cv2.line(map_img, (0, i), (width, i), (30, 30, 30), 1)
        
        # Add info
        cv2.putText(map_img, f"Points: {len(self.trajectory)}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        pos = self.current_pose
        cv2.putText(map_img, f"X: {pos[0]:.1f} Z: {pos[2]:.1f}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(map_img, f"Scale: {scale:.1f}", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return map_img
    
    def reset(self):
        """Reset tracking state"""
        self.prev_gray = None
        self.prev_points = None
        self.trajectory = []
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.trails = []
        self.frame_count = 0
        self.scale = 1.0


class SpatialStreamReceiverGUI:
    def __init__(self, root):
        
        self.root = root
        self.root.title("Ultra-Lightweight Visual Odometry Receiver")
        self.root.geometry("1400x800")
        self.root.configure(bg='#1e1e1e')
        #start_display(root)
        
        # Enable OpenCV optimizations
        cv2.setUseOptimized(True)
        
        # Connection variables
        self.host = '100.89.208.16'
        self.port = 8888
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.receiving_thread = None
        
        # Visual odometry
        self.vo = None
        self.is_tracking = False
        self.tracking_thread = None
        
        # Performance settings
        self.downscale_factor = 0.75
        self.max_features = 200
        
        # Frame data
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        self.fps = 0
        self.tracking_fps = 0
        self.last_fps_time = time.time()
        self.last_tracking_fps_time = time.time()
        self.tracking_frame_count = 0
        self.bytes_received = 0
        self.total_frames = 0
        
        # Frame queues
        self.frame_queue = Queue(maxsize=2)
        self.tracking_queue = Queue(maxsize=30)  # Buffer for tracking
        self.map_image_queue = Queue(maxsize=2)
        
        self.setup_gui()
        self.update_displays()
        
    def handle_voice_command(self, text):
        """Handle voice commands safely from another thread"""
        cmd = text.upper()

        def ui(action):
            self.root.after(0, action)

        if "START CAMERA" in cmd:
            ui(self.toggle_server)

        elif "STOP CAMERA" in cmd:
            ui(self.stop_server)

        elif "START TRACKING" in cmd:
            ui(self.start_tracking)

        elif "STOP TRACKING" in cmd:
            ui(self.stop_tracking)

        elif "RESET" in cmd or "RESET TRAJECTORY" in cmd:
            ui(self.reset_trajectory)

        elif "EXIT" in cmd or "QUIT" in cmd or "CLOSE PROGRAM" in cmd:
            ui(self.on_closing)

    def setup_gui(self):
        """Setup the GUI layout"""
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Top control panel
        control_frame = tk.Frame(self.root, bg='#2d2d2d', pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        # Server IP
        tk.Label(control_frame, text="Server IP:", bg='#2d2d2d', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        self.ip_entry = tk.Entry(control_frame, width=20, bg='#3a3a3a', fg='white', font=('Arial', 10))
        self.ip_entry.insert(0, self.host)
        self.ip_entry.pack(side=tk.LEFT, padx=5)
        
        # Port
        tk.Label(control_frame, text="Port:", bg='#2d2d2d', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        self.port_entry = tk.Entry(control_frame, width=8, bg='#3a3a3a', fg='white', font=('Arial', 10))
        self.port_entry.insert(0, str(self.port))
        self.port_entry.pack(side=tk.LEFT, padx=5)
        
        # Start Server button
        self.start_button = tk.Button(
            control_frame, text="Start Server", command=self.toggle_server,
            bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
            padx=20, relief=tk.FLAT, cursor='hand2'
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        # Start Tracking button
        self.start_tracking_btn = tk.Button(
            control_frame, text="Start Tracking", command=self.start_tracking,
            bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
            padx=20, relief=tk.FLAT, cursor='hand2', state='disabled'
        )
        self.start_tracking_btn.pack(side=tk.LEFT, padx=5)
        
        # Stop Tracking button
        self.stop_tracking_btn = tk.Button(
            control_frame, text="Stop Tracking", command=self.stop_tracking,
            bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
            padx=20, relief=tk.FLAT, cursor='hand2', state='disabled'
        )
        self.stop_tracking_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        self.reset_btn = tk.Button(
            control_frame, text="Reset Trajectory", command=self.reset_trajectory,
            bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'),
            padx=20, relief=tk.FLAT, cursor='hand2', state='disabled'
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            control_frame, text="● Disconnected", bg='#2d2d2d', fg='#ff5252',
            font=('Arial', 10, 'bold')
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Performance controls
        perf_frame = tk.Frame(self.root, bg='#2d2d2d', pady=5)
        perf_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        tk.Label(perf_frame, text="Settings:", bg='#2d2d2d', fg='#FFC107', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=10)
        
        # Scale control
        tk.Label(perf_frame, text="Scale:", bg='#2d2d2d', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.scale_var = tk.StringVar(value="0.75")
        scale_options = ["0.5", "0.75", "1.0"]
        self.scale_menu = ttk.Combobox(perf_frame, textvariable=self.scale_var, values=scale_options, width=5, state='readonly')
        self.scale_menu.pack(side=tk.LEFT, padx=5)
        self.scale_menu.bind('<<ComboboxSelected>>', lambda e: setattr(self, 'downscale_factor', float(self.scale_var.get())))
        
        # Max features
        tk.Label(perf_frame, text="Max Features:", bg='#2d2d2d', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.features_var = tk.StringVar(value="300")
        features_options = ["100", "200", "300", "500"]
        self.features_menu = ttk.Combobox(perf_frame, textvariable=self.features_var, values=features_options, width=5, state='readonly')
        self.features_menu.pack(side=tk.LEFT, padx=5)
        self.features_menu.bind('<<ComboboxSelected>>', self.update_max_features)
        
        # Motion sensitivity
        tk.Label(perf_frame, text="Motion Sensitivity:", bg='#2d2d2d', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.sensitivity_var = tk.StringVar(value="0.3")
        sensitivity_options = ["0.1", "0.2", "0.3", "0.5", "0.7"]
        self.sensitivity_menu = ttk.Combobox(perf_frame, textvariable=self.sensitivity_var, values=sensitivity_options, width=5, state='readonly')
        self.sensitivity_menu.pack(side=tk.LEFT, padx=5)
        self.sensitivity_menu.bind('<<ComboboxSelected>>', self.update_sensitivity)
        
        # Quality indicator
        self.quality_label = tk.Label(
            perf_frame, text="Quality: --", bg='#2d2d2d', fg='#03A9F4', font=('Arial', 9)
        )
        self.quality_label.pack(side=tk.LEFT, padx=10)
        
        # Stats panel
        stats_frame = tk.Frame(self.root, bg='#2d2d2d', pady=5)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        self.fps_label = tk.Label(stats_frame, text="Video FPS: 0.0", bg='#2d2d2d', fg='#4CAF50', font=('Arial', 10))
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        self.tracking_fps_label = tk.Label(stats_frame, text="Tracking FPS: 0.0", bg='#2d2d2d', fg='#2196F3', font=('Arial', 10))
        self.tracking_fps_label.pack(side=tk.LEFT, padx=10)
        
        self.frames_label = tk.Label(stats_frame, text="Frames: 0", bg='#2d2d2d', fg='white', font=('Arial', 10))
        self.frames_label.pack(side=tk.LEFT, padx=10)
        
        self.features_label = tk.Label(stats_frame, text="Features: 0", bg='#2d2d2d', fg='#9C27B0', font=('Arial', 10))
        self.features_label.pack(side=tk.LEFT, padx=10)
        
        self.bandwidth_label = tk.Label(stats_frame, text="Bandwidth: 0 KB/s", bg='#2d2d2d', fg='white', font=('Arial', 10))
        self.bandwidth_label.pack(side=tk.LEFT, padx=10)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Video feed frame (Left)
        video_frame = tk.Frame(content_frame, bg='#0d0d0d', relief=tk.SUNKEN, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        video_label = tk.Label(video_frame, text="Camera Feed (Real-time)", 
                              bg='#0d0d0d', fg='#888888', font=('Arial', 14))
        video_label.pack(pady=5)
        
        self.video_canvas = tk.Label(video_frame, bg='#0d0d0d', 
                                     text="Disconnected", fg='#ff0000',
                                     font=('Arial', 16, 'bold'))
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tracking frame (Right)
        tracking_frame = tk.Frame(content_frame, bg='#0d0d0d', relief=tk.SUNKEN, bd=2)
        tracking_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        tracking_label = tk.Label(tracking_frame, text="Trajectory Map", 
                                 bg='#0d0d0d', fg='#888888', font=('Arial', 14))
        tracking_label.pack(pady=5)
        
        self.tracking_canvas = tk.Label(tracking_frame, bg='#0d0d0d',
                                        text="No tracking active", fg='#888888',
                                        font=('Arial', 14))
        self.tracking_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def update_max_features(self, event=None):
        """Update max features"""
        self.max_features = int(self.features_var.get())
        if self.vo is not None:
            self.vo.feature_params['maxCorners'] = self.max_features
    
    def update_sensitivity(self, event=None):
        """Update motion sensitivity/damping"""
        sensitivity = float(self.sensitivity_var.get())
        if self.vo is not None:
            self.vo.motion_damping = sensitivity
    
    def toggle_server(self):
        """Start or stop the server"""
        if not self.running:
            self.start_server()
        else:
            self.stop_server()
    
    def start_server(self):
        """Start listening for connections"""
        try:
            self.host = self.ip_entry.get()
            self.port = int(self.port_entry.get())
            
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            
            self.running = True
            self.start_button.config(text="Stop Server", bg='#f44336')
            self.status_label.config(text="● Waiting for connection...", fg='#FFC107')
            
            threading.Thread(target=self.accept_connection, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start server: {e}")
            self.running = False
    
    def accept_connection(self):
        """Accept incoming connection"""
        try:
            print(f"Server listening on {self.host}:{self.port}")
            self.client_socket, client_address = self.server_socket.accept()
            print(f"Connected to {client_address}")
            
            self.root.after(0, lambda: self.status_label.config(
                text=f"● Connected to {client_address[0]}", fg='#4CAF50'
            ))
            self.root.after(0, lambda: self.start_tracking_btn.config(state='normal'))
            
            self.receiving_thread = threading.Thread(target=self.receive_frames, daemon=True)
            self.receiving_thread.start()
            
        except Exception as e:
            if self.running:
                print(f"Connection error: {e}")
                self.root.after(0, lambda: self.stop_server())
    
    def receive_exact(self, num_bytes):
        """Receive exact number of bytes from socket"""
        data = b''
        while len(data) < num_bytes and self.running:
            try:
                chunk = self.client_socket.recv(min(num_bytes - len(data), 8192))
                if not chunk:
                    raise ConnectionError("Connection closed")
                data += chunk
            except socket.timeout:
                continue
        return data
    
    def receive_frames(self):
        """Receive and process frames"""
        self.last_bandwidth_time = time.time()
        self.bandwidth_bytes = 0
        
        try:
            while self.running:
                size_bytes = self.receive_exact(4)
                if len(size_bytes) < 4:
                    break
                    
                frame_size = struct.unpack('!I', size_bytes)[0]
                jpeg_data = self.receive_exact(frame_size)
                if len(jpeg_data) < frame_size:
                    break
                
                self.bandwidth_bytes += len(jpeg_data) + 4
                
                nparr = np.frombuffer(jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    
                    # Put in tracking queue if tracking is active
                    if self.is_tracking and not self.tracking_queue.full():
                        self.tracking_queue.put(frame.copy())
                    
                    # Put frame in queue for real-time display
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    
                    with self.frame_lock:
                        self.current_frame = frame
                        self.total_frames += 1
                    
                    self.update_fps()
                    
                    # Update bandwidth
                    current_time = time.time()
                    if current_time - self.last_bandwidth_time >= 1.0:
                        bandwidth_kbps = (self.bandwidth_bytes / 1024) / (current_time - self.last_bandwidth_time)
                        self.root.after(0, lambda bw=bandwidth_kbps: self.bandwidth_label.config(
                            text=f"Bandwidth: {bw:.1f} KB/s"
                        ))
                        self.bandwidth_bytes = 0
                        self.last_bandwidth_time = current_time
                
        except Exception as e:
            print(f"Error receiving frames: {e}")
        finally:
            if self.running:
                self.root.after(0, lambda: self.stop_server())
    
    def update_fps(self):
        """Update video FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time
            self.root.after(0, lambda: self.fps_label.config(text=f"Video FPS: {self.fps:.1f}"))
            self.root.after(0, lambda: self.frames_label.config(text=f"Frames: {self.total_frames}"))
    
    def update_tracking_fps(self):
        """Update tracking FPS calculation"""
        self.tracking_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_tracking_fps_time
        
        if elapsed >= 1.0:
            self.tracking_fps = self.tracking_frame_count / elapsed
            self.tracking_frame_count = 0
            self.last_tracking_fps_time = current_time
            self.root.after(0, lambda: self.tracking_fps_label.config(
                text=f"Tracking FPS: {self.tracking_fps:.1f}"
            ))
    
    def tracking_loop(self):
        """Main tracking loop"""
        print("Visual odometry tracking started")
        
        while self.is_tracking:
            try:
                # Get next frame from queue
                try:
                    frame = self.tracking_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Downscale if needed
                if self.downscale_factor != 1.0:
                    h, w = frame.shape[:2]
                    new_w = int(w * self.downscale_factor)
                    new_h = int(h * self.downscale_factor)
                    frame = cv2.resize(frame, (new_w, new_h))
                
                # Process frame
                tracked_frame, quality, num_features = self.vo.process_frame(frame)
                
                # Update quality display
                self.root.after(0, lambda q=quality: self.quality_label.config(
                    text=f"Quality: {q:.1f}%"
                ))
                self.root.after(0, lambda n=num_features: self.features_label.config(
                    text=f"Features: {n}"
                ))
                
                # Generate trajectory map
                map_img = self.vo.draw_trajectory_map()
                
                # Put processed images in queues
                if not self.map_image_queue.full():
                    # Combine tracked frame and map side by side for display
                    h1, w1 = tracked_frame.shape[:2]
                    h2, w2 = map_img.shape[:2]
                    
                    # Resize map to match frame height
                    map_resized = cv2.resize(map_img, (int(w2 * h1 / h2), h1))
                    
                    # Concatenate
                    combined = np.hstack([tracked_frame, map_resized])
                    self.map_image_queue.put(combined)
                
                self.update_tracking_fps()
                
            except Exception as e:
                print(f"Tracking error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        print("Visual odometry tracking stopped")
    
    def start_tracking(self):
        """Start visual odometry tracking"""
        if not self.running:
            self.update_status("Connect server first", "#ff0000")
            return
        
        with self.frame_lock:
            if self.current_frame is None:
                self.update_status("Waiting for frames...", "#ffaa00")
                return
        
        # Initialize VO
        with self.frame_lock:
            h, w = self.current_frame.shape[:2]
        
        focal = w * 0.8
        pp = (w / 2, h / 2)
        self.vo = LightweightVisualOdometry(focal_length=focal, pp=pp)
        self.vo.feature_params['maxCorners'] = self.max_features
        
        # Clear tracking queue
        while not self.tracking_queue.empty():
            try:
                self.tracking_queue.get_nowait()
            except Empty:
                break
        
        self.is_tracking = True
        self.start_tracking_btn.config(state='disabled')
        self.stop_tracking_btn.config(state='normal')
        self.reset_btn.config(state='normal')
        self.update_status("Tracking Active", "#00ff00")
        
        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self.tracking_loop, daemon=True)
        self.tracking_thread.start()
    
    def stop_tracking(self):
        """Stop tracking"""
        self.is_tracking = False
        self.start_tracking_btn.config(state='normal' if self.running else 'disabled')
        self.stop_tracking_btn.config(state='disabled')
        self.reset_btn.config(state='disabled')
        self.update_status("Tracking Stopped", "#ffaa00")
    
    def reset_trajectory(self):
        """Reset the trajectory"""
        if self.vo is not None:
            self.vo.reset()
            print("Trajectory reset")
    
    def update_displays(self):
        """Update video and tracking displays"""
        # Update video display
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                self.display_frame(frame, self.video_canvas)
        except Empty:
            pass
        
        # Update tracking display
        try:
            if not self.map_image_queue.empty():
                img = self.map_image_queue.get_nowait()
                self.display_image(img, self.tracking_canvas)
        except Empty:
            pass
        
        self.root.after(33, self.update_displays)
    
    def display_frame(self, frame, canvas):
        """Display frame on canvas"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 680
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else 600
        
        h, w = frame_rgb.shape[:2]
        scale = min(canvas_width/w, canvas_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        canvas.configure(image=imgtk, text="")
        canvas.image = imgtk
    
    def display_image(self, img, canvas):
        """Display image on canvas"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 600
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else 400
        
        h, w = img_rgb.shape[:2]
        scale = min(canvas_width/w, canvas_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        img_pil = Image.fromarray(img_resized)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        
        canvas.configure(image=imgtk, text="")
        canvas.image = imgtk
    
    def update_status(self, message, color):
        """Update status label"""
        self.status_label.config(text=f"● {message}", fg=color)
    
    def stop_server(self):
        """Stop the server"""
        if self.is_tracking:
            self.stop_tracking()
        
        self.running = False
        
        try:
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()
        except:
            pass
        
        self.client_socket = None
        self.server_socket = None
        
        self.start_button.config(text="Start Server", bg='#4CAF50')
        self.start_tracking_btn.config(state='disabled')
        self.stop_tracking_btn.config(state='disabled')
        self.reset_btn.config(state='disabled')
        self.status_label.config(text="● Disconnected", fg='#ff5252')
        
        self.video_canvas.configure(image='', text="Disconnected", fg="#ff0000")
        self.video_canvas.image = None
        self.tracking_canvas.configure(image='', text="No tracking active", fg="#888888")
        self.tracking_canvas.image = None
        
        print("Server stopped")
    
    def on_closing(self):
        """Handle window close"""
        if self.is_tracking:
            self.stop_tracking()
        self.stop_server()
        self.root.destroy()
def continuous_listen_and_print(app):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)

        while True:
            try:
                audio = recognizer.listen(source, phrase_time_limit=10)
                text = recognizer.recognize_google(audio)
                print(f"Heard: {text}")

                if "EAGLE" in text.upper():
                    app.handle_voice_command(text)

            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                break


def main():
    # Wait for double clap first
    listen_for_claps()
    
    # start all threads after claps detected
    stop_event = threading.Event()

    # Start controller thread
    threading.Thread(
        target=controller_input_thread,
        args=(stop_event, controller_state, state_lock),
        daemon=True
    ).start()

    # Start GUI
    root = tk.Tk()
    app = SpatialStreamReceiverGUI(root)

    # Start voice control thread
    threading.Thread(
        target=continuous_listen_and_print,
        args=(app,),
        daemon=True
    ).start()
 
    # Run the GUI main loop
    root.mainloop()

    
    # Clean up when GUI closes
    stop_event.set()




if __name__ == "__main__":
    main()
