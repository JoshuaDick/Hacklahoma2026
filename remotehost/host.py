#!/usr/bin/env python3
"""
Spatial Streamer Receiver with GUI
Receives camera feed with heat map overlay from Android device
"""

import socket
import struct
import cv2
import numpy as np
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class SpatialStreamReceiverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spatial Stream Receiver")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Connection variables
        self.host = '100.89.208.16'
        self.port = 8888
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.receiving_thread = None
        
        # Frame data
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.bytes_received = 0
        self.total_frames = 0
        
        self.setup_gui()
        self.update_display()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', padding=6, relief="flat", background="#4a4a4a")
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TEntry', fieldbackground='#3a3a3a', foreground='white')
        
        # Top control panel
        control_frame = tk.Frame(self.root, bg='#2b2b2b', pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        # Server IP
        tk.Label(control_frame, text="Server IP:", bg='#2b2b2b', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        self.ip_entry = tk.Entry(control_frame, width=20, bg='#3a3a3a', fg='white', font=('Arial', 10))
        self.ip_entry.insert(0, self.host)
        self.ip_entry.pack(side=tk.LEFT, padx=5)
        
        # Port
        tk.Label(control_frame, text="Port:", bg='#2b2b2b', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        self.port_entry = tk.Entry(control_frame, width=8, bg='#3a3a3a', fg='white', font=('Arial', 10))
        self.port_entry.insert(0, str(self.port))
        self.port_entry.pack(side=tk.LEFT, padx=5)
        
        # Start button
        self.start_button = tk.Button(
            control_frame, 
            text="Start Server", 
            command=self.toggle_server,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = tk.Label(
            control_frame, 
            text="● Disconnected", 
            bg='#2b2b2b', 
            fg='#ff5252',
            font=('Arial', 10, 'bold')
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Stats panel
        stats_frame = tk.Frame(self.root, bg='#2b2b2b', pady=5)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        self.fps_label = tk.Label(stats_frame, text="FPS: 0.0", bg='#2b2b2b', fg='#4CAF50', font=('Arial', 10))
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        self.frames_label = tk.Label(stats_frame, text="Frames: 0", bg='#2b2b2b', fg='white', font=('Arial', 10))
        self.frames_label.pack(side=tk.LEFT, padx=10)
        
        self.bandwidth_label = tk.Label(stats_frame, text="Bandwidth: 0 KB/s", bg='#2b2b2b', fg='white', font=('Arial', 10))
        self.bandwidth_label.pack(side=tk.LEFT, padx=10)
        
        self.resolution_label = tk.Label(stats_frame, text="Resolution: N/A", bg='#2b2b2b', fg='white', font=('Arial', 10))
        self.resolution_label.pack(side=tk.LEFT, padx=10)
        
        # Video display canvas
        self.canvas = tk.Canvas(self.root, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Instructions
        instructions = (
            "Instructions:\n"
            "1. Enter your Tailscale IP (find with: tailscale ip)\n"
            "2. Click 'Start Server' to listen for connections\n"
            "3. Enter this IP in the Android app and click 'Connect'\n"
            "4. Heat map overlay shows distance (blue=far, red=close)"
        )
        self.instructions_label = tk.Label(
            self.root, 
            text=instructions,
            bg='#2b2b2b', 
            fg='#888888',
            font=('Arial', 9),
            justify=tk.LEFT,
            pady=5
        )
        self.instructions_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
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
            
            # Start accepting connections in a separate thread
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
                text=f"● Connected to {client_address[0]}", 
                fg='#4CAF50'
            ))
            
            # Start receiving frames
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
                chunk = self.client_socket.recv(min(num_bytes - len(data), 4096))
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
                # Read frame size (4 bytes)
                size_bytes = self.receive_exact(4)
                if len(size_bytes) < 4:
                    break
                    
                frame_size = struct.unpack('!I', size_bytes)[0]
                
                # Read JPEG data
                jpeg_data = self.receive_exact(frame_size)
                if len(jpeg_data) < frame_size:
                    break
                
                # Update bandwidth stats
                self.bandwidth_bytes += len(jpeg_data) + 4
                
                # Decode JPEG
                nparr = np.frombuffer(jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Convert BGR to RGB for Tkinter
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    with self.frame_lock:
                        self.current_frame = frame
                        self.total_frames += 1
                    
                    # Update FPS
                    self.update_fps()
                    
                    # Update bandwidth every second
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
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time
            
            # Update GUI
            self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {self.fps:.1f}"))
            self.root.after(0, lambda: self.frames_label.config(text=f"Frames: {self.total_frames}"))
    
    def update_display(self):
        """Update the video display"""
        with self.frame_lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
                
                # Update resolution label
                h, w = frame.shape[:2]
                self.resolution_label.config(text=f"Resolution: {w}x{h}")
                
                # Resize to fit canvas while maintaining aspect ratio
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    # Calculate scaling
                    scale_w = canvas_width / w
                    scale_h = canvas_height / h
                    frame_resized = cv2.resize(frame, (canvas_width, canvas_height))

                    
                    # Convert to PhotoImage
                    img = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image=img)
                    
                    # Update canvas
                    self.canvas.delete("all")
                    x = (canvas_width - canvas_width) // 2
                    y = (canvas_height - canvas_height) // 2
                    self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
                    self.canvas.image = photo  # Keep a reference
        
        # Schedule next update
        self.root.after(33, self.update_display)  # ~30 FPS display update
    
    def stop_server(self):
        """Stop the server and close connections"""
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
        self.status_label.config(text="● Disconnected", fg='#ff5252')
        self.fps_label.config(text="FPS: 0.0")
        
        print("Server stopped")
    
    def on_closing(self):
        """Handle window close event"""
        self.stop_server()
        self.root.destroy()

def main():
    print("="*60)
    print("Spatial Streamer Receiver - GUI Version")
    print("="*60)
    print("\nStarting GUI...")
    
    root = tk.Tk()
    app = SpatialStreamReceiverGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()