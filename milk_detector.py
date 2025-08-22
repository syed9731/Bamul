#!/usr/bin/env python3
"""
Milk Packet Detector for Raspberry Pi
Detects milk packets crossing a counting line on conveyor belt
"""

import cv2
import numpy as np
import time
import yaml
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

class MilkPacketDetector:
    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize camera with simpler configuration
        self.picam2 = Picamera2()
        
        # Use basic preview configuration
        preview_config = self.picam2.create_preview_configuration(
            main={"size": (self.config['resolution']['width'], self.config['resolution']['height'])}
        )
        self.picam2.configure(preview_config)
        self.picam2.start()
        
        # Set camera controls using the correct API
        try:
            self.picam2.set_controls({"ExposureTime": self.config['camera']['exposure_time']})
            self.picam2.set_controls({"AnalogueGain": self.config['camera']['analogue_gain']})
        except Exception as e:
            print(f"Warning: Could not set camera controls: {e}")
            print("Using default camera settings")
        
        # Initialize TFLite interpreter
        self.interpreter = tflite.Interpreter(model_path=self.config['model']['path'])
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Initialize variables
        self.packet_count = 0
        self.last_detection_time = 0
        self.detection_cooldown = 1.0 / self.config['fps']  # Minimum time between detections
        
        # Calculate counting line position (horizontal line across middle of frame)
        self.counting_line_y = self.config['resolution']['height'] // 2
        
        # Initialize FPS calculation
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
    def preprocess_frame(self, frame):
        """Preprocess frame for TFLite model"""
        # Resize to model input size
        input_size = (self.config['model']['input_width'], self.config['model']['input_height'])
        resized = cv2.resize(frame, input_size)
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def detect_packets(self, frame):
        """Run inference on frame to detect milk packets"""
        # Preprocess frame
        input_data = self.preprocess_frame(frame)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        detections = []
        for i in range(len(scores)):
            if scores[i] > self.config['detection']['confidence_threshold']:
                # Convert normalized coordinates to pixel coordinates
                y1, x1, y2, x2 = boxes[i]
                x1 = int(x1 * self.config['resolution']['width'])
                y1 = int(y1 * self.config['resolution']['height'])
                x2 = int(x2 * self.config['resolution']['width'])
                y2 = int(y2 * self.config['resolution']['height'])
                
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'score': scores[i],
                    'class': int(classes[i])
                })
        
        return detections
    
    def check_crossing_line(self, detections):
        """Check if any detected packet crosses the counting line"""
        current_time = time.time()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            
            # Check if packet center crosses the counting line
            packet_center_y = (y1 + y2) // 2
            
            # Simple crossing detection: if packet center is near the line
            if abs(packet_center_y - self.counting_line_y) < 20:  # 20 pixel tolerance
                if current_time - self.last_detection_time > self.detection_cooldown:
                    self.packet_count += 1
                    self.last_detection_time = current_time
                    print(f"Milk packet detected! Count: {self.packet_count}")
                    break
    
    def draw_overlay(self, frame, detections):
        """Draw detection boxes and counting information on frame"""
        # Draw counting line
        cv2.line(frame, (0, self.counting_line_y), 
                (self.config['resolution']['width'], self.counting_line_y), 
                (0, 255, 0), 2)
        
        # Draw detection boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            score = detection['score']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw confidence score
            cv2.putText(frame, f"{score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw packet count
        cv2.putText(frame, f"Count: {self.packet_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def run(self):
        """Main detection loop"""
        print("Starting milk packet detection...")
        print(f"Counting line at Y position: {self.counting_line_y}")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Convert from BGR to RGB (TFLite expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection
                detections = self.detect_packets(frame_rgb)
                
                # Check for line crossing
                self.check_crossing_line(detections)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Draw overlay
                frame_with_overlay = self.draw_overlay(frame, detections)
                
                # Display frame
                cv2.imshow('Milk Packet Detector', frame_with_overlay)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Control FPS
                time.sleep(1.0 / self.config['fps'])
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.picam2.stop()
            cv2.destroyAllWindows()
            print(f"Final packet count: {self.packet_count}")

if __name__ == "__main__":
    detector = MilkPacketDetector()
    detector.run() 