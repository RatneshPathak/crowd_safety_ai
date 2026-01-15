"""
Enhanced main.py that works with your existing code structure
"""

import cv2
import numpy as np
import time
from pathlib import Path

# Your existing function-based imports
from video_loader import VideoLoader
from preprocessing import Preprocessor
from visualizer import DashboardVisualizer
from anomaly_detection import AnomalyDetector
import density_estimation
import motion_analysis
import risk_classifier

# Import the new dashboard
from dashboard import CrowdSafetyDashboard

import config


class EnhancedCrowdSafetySystem:
    def __init__(self):
        """Initialize all components including dashboard"""
        self.preprocessor = Preprocessor()
        self.visualizer = DashboardVisualizer()
        self.anomaly_detector = AnomalyDetector()
        
        # Initialize dashboard with 150 frames of history
        self.dashboard = CrowdSafetyDashboard(max_history=150)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.prev_time = time.time()
        
        # History tracking for metrics
        self.density_history = []
        self.risk_history = []
        self.motion_history = []
        
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.prev_time)
            self.frame_count = 0
            self.prev_time = current_time
            
        return self.fps
    
    def generate_alerts(self, density, risk_score, anomaly_detected, motion_magnitude):
        """Generate alerts based on current conditions"""
        # Clear old alerts
        self.dashboard.clear_old_alerts(max_age=5.0)
        
        # Convert risk_score (0-1) for comparisons
        risk_normalized = risk_score if isinstance(risk_score, (int, float)) else 0.5
        
        # Critical risk alert
        if risk_normalized > 0.85:
            self.dashboard.add_alert(
                'CRITICAL RISK', 
                'high',
                f'EMERGENCY: Risk level at {risk_normalized:.0%}!'
            )
        elif risk_normalized > 0.7:
            self.dashboard.add_alert(
                'High Risk',
                'high',
                f'High crowd risk detected: {risk_normalized:.0%}'
            )
        elif risk_normalized > 0.5:
            self.dashboard.add_alert(
                'Moderate Risk',
                'medium',
                f'Caution advised: Risk at {risk_normalized:.0%}'
            )
        
        # Density alerts
        if density > 0.8:
            self.dashboard.add_alert(
                'High Density',
                'high',
                f'Severe crowding: {density:.0%} capacity'
            )
        
        # Motion alerts
        if motion_magnitude > 15.0:  # Adjusted threshold for your motion values
            self.dashboard.add_alert(
                'High Motion',
                'medium',
                f'Intense movement: {motion_magnitude:.1f} px/frame'
            )
        
        # Anomaly alert
        if anomaly_detected:
            self.dashboard.add_alert(
                'Anomaly Detected',
                'medium',
                'Unusual crowd behavior pattern'
            )
    
    def normalize_risk(self, risk_str, density, motion):
        """Convert risk string to normalized 0-1 value"""
        if isinstance(risk_str, (int, float)):
            return risk_str
        
        # Convert string risk to numerical
        if risk_str == "HIGH":
            base = 0.75
        elif risk_str == "MEDIUM":
            base = 0.5
        else:  # LOW
            base = 0.25
        
        # Adjust based on density and motion
        adjustment = (density * 0.3 + min(motion / 20.0, 1.0) * 0.2)
        return min(base + adjustment, 1.0)
    
    def process_video(self, video_path, output_path=None, display=True):
        """Process video with enhanced dashboard visualization"""
        # Load video - initialize VideoLoader with path
        try:
            video_loader = VideoLoader(video_path)
        except Exception as e:
            print(f"Error: Could not load video from {video_path}")
            print(f"Details: {e}")
            return
        
        # Get video properties from the VideoLoader's cap
        cap = video_loader.cap
        fps_original = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps_original}")
        print(f"Total frames: {total_frames}")
        print("-" * 50)
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Dashboard output is 1920x1080
            out = cv2.VideoWriter(output_path, fourcc, fps_original, (1920, 1080))
        
        frame_num = 0
        prev_gray = None
        model_accuracy = 92.5  # Mock accuracy for visualization
        
        try:
            while True:
                # Use VideoLoader's read method
                ret, frame = video_loader.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Preprocess frame
                gray_frame = self.preprocessor.process(frame)
                
                # Density estimation using your function
                rows, cols = 10, 10
                density_map = density_estimation.estimate_density(gray_frame, rows, cols)
                density_value = np.mean(density_map)
                
                # Motion analysis using your function
                if prev_gray is not None:
                    motion_magnitude = motion_analysis.compute_motion(prev_gray, gray_frame)
                else:
                    motion_magnitude = 0.0
                
                prev_gray = gray_frame.copy()
                
                # Anomaly detection
                anomaly_score = self.anomaly_detector.compute_score(motion_magnitude, density_value)
                anomaly_detected = anomaly_score > 0.7  # Threshold for anomaly
                
                # Risk classification using your function
                # Create a combined score for risk classification
                risk_score = (density_value * 0.6 + min(motion_magnitude / 20.0, 1.0) * 0.4)
                risk_str = risk_classifier.classify_risk(risk_score)
                risk_normalized = self.normalize_risk(risk_str, density_value, motion_magnitude)
                
                # Update history
                self.density_history.append(density_value)
                self.risk_history.append(risk_normalized)
                self.motion_history.append(motion_magnitude)
                
                # Keep history length manageable
                if len(self.density_history) > 150:
                    self.density_history.pop(0)
                    self.risk_history.pop(0)
                    self.motion_history.pop(0)
                
                # Calculate FPS
                current_fps = self.calculate_fps()
                
                # Update dashboard metrics
                self.dashboard.update_metrics(
                    density=density_value,
                    risk_level=risk_normalized,
                    motion_magnitude=motion_magnitude / 20.0,  # Normalize to 0-1
                    anomaly_detected=anomaly_detected,
                    fps=current_fps
                )
                
                # Generate alerts
                self.generate_alerts(
                    density_value, 
                    risk_normalized, 
                    anomaly_detected,
                    motion_magnitude
                )
                
                # Create visualization using your existing visualizer
                vis_frame = self.visualizer.create_pro_dashboard(
                    frame, 
                    density_value,
                    motion_magnitude,
                    risk_normalized,
                    current_fps,
                    self.density_history,
                    self.risk_history,
                    model_accuracy
                )
                
                # Render complete dashboard with visualization
                dashboard_frame = self.dashboard.render_dashboard(vis_frame, model_accuracy)
                
                # Display
                if display:
                    # Resize for display if too large
                    display_frame = cv2.resize(dashboard_frame, (1280, 720))
                    cv2.imshow('Crowd Safety AI Dashboard', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nStopping video processing...")
                        break
                    elif key == ord('p'):
                        print("Paused. Press any key to continue...")
                        cv2.waitKey(0)
                
                # Write to output
                if output_path:
                    out.write(dashboard_frame)
                
                # Progress indicator
                if frame_num % 30 == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | "
                          f"FPS: {current_fps:.1f} | "
                          f"Risk: {risk_str} ({risk_normalized:.2f})", end='\r')
        
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            video_loader.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 50)
            print("PROCESSING COMPLETE")
            print(f"Total frames processed: {frame_num}")
            print(f"Average FPS: {np.mean(self.dashboard.fps_history) if self.dashboard.fps_history else 0:.2f}")
            print(f"High risk frames: {self.dashboard.high_risk_frames}")
            print(f"Anomalies detected: {self.dashboard.anomaly_count}")
            print("=" * 50)


def main():
    """Main execution function"""
    # Initialize system
    system = EnhancedCrowdSafetySystem()
    
    # Input video path
    video_path = getattr(config, 'VIDEO_PATH', 'data/videos/merged_crowd_demo.mp4')
    
    # Output path
    output_dir = Path(getattr(config, 'OUTPUT_DIR', 'data/outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "crowd_safety_dashboard_output.mp4"
    
    print("=" * 50)
    print("CROWD SAFETY AI - ENHANCED DASHBOARD SYSTEM")
    print("=" * 50)
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print("=" * 50)
    print("\nControls:")
    print("  Q - Quit")
    print("  P - Pause/Resume")
    print("=" * 50)
    print()
    
    # Process video
    system.process_video(
        video_path=video_path,
        output_path=str(output_path),
        display=True
    )
    
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()