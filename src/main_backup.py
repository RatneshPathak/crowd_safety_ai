"""
Enhanced main.py with integrated dashboard
Replace your existing main.py with this file
"""

import cv2
import numpy as np
import time
from pathlib import Path

# Your existing imports
from src.video_loader import VideoLoader
from src.preprocessing import Preprocessor
from src.density_estimation import DensityEstimator
from src.motion_analysis import MotionAnalyzer
from src.anomaly_detection import AnomalyDetector
from src.risk_classifier import RiskClassifier
from src.visualizer import Visualizer

# Import the new dashboard (save the previous artifact as src/dashboard.py)
from src.dashboard import CrowdSafetyDashboard

import config


class EnhancedCrowdSafetySystem:
    def __init__(self):
        """Initialize all components including dashboard"""
        self.video_loader = VideoLoader()
        self.preprocessor = Preprocessor()
        self.density_estimator = DensityEstimator()
        self.motion_analyzer = MotionAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.risk_classifier = RiskClassifier()
        self.visualizer = Visualizer()
        
        # Initialize dashboard with 150 frames of history
        self.dashboard = CrowdSafetyDashboard(max_history=150)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.prev_time = time.time()
        
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.prev_time)
            self.frame_count = 0
            self.prev_time = current_time
            
        return self.fps
    
    def generate_alerts(self, density, risk, anomaly_detected, motion_magnitude):
        """Generate alerts based on current conditions"""
        # Clear old alerts
        self.dashboard.clear_old_alerts(max_age=5.0)
        
        # Critical risk alert
        if risk > 0.85:
            self.dashboard.add_alert(
                'CRITICAL RISK', 
                'high',
                f'EMERGENCY: Risk level at {risk:.0%}!'
            )
        elif risk > 0.7:
            self.dashboard.add_alert(
                'High Risk',
                'high',
                f'High crowd risk detected: {risk:.0%}'
            )
        elif risk > 0.5:
            self.dashboard.add_alert(
                'Moderate Risk',
                'medium',
                f'Caution advised: Risk at {risk:.0%}'
            )
        
        # Density alerts
        if density > 0.8:
            self.dashboard.add_alert(
                'High Density',
                'high',
                f'Severe crowding: {density:.0%} capacity'
            )
        
        # Motion alerts
        if motion_magnitude > 0.75:
            self.dashboard.add_alert(
                'High Motion',
                'medium',
                'Intense crowd movement detected'
            )
        
        # Anomaly alert
        if anomaly_detected:
            self.dashboard.add_alert(
                'Anomaly Detected',
                'medium',
                'Unusual crowd behavior pattern'
            )
    
    def process_video(self, video_path, output_path=None, display=True):
        """Process video with enhanced dashboard visualization"""
        # Load video
        cap = self.video_loader.load_video(video_path)
        if cap is None:
            print(f"Error: Could not load video from {video_path}")
            return
        
        # Get video properties
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
        prev_frame = None
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Preprocess frame
                processed_frame = self.preprocessor.preprocess(frame)
                
                # Density estimation
                density_map = self.density_estimator.estimate_density(processed_frame)
                density_value = np.mean(density_map)
                
                # Motion analysis
                if prev_frame is not None:
                    motion_vectors = self.motion_analyzer.analyze_motion(
                        prev_frame, processed_frame
                    )
                    motion_magnitude = self.motion_analyzer.get_motion_magnitude(
                        motion_vectors
                    )
                else:
                    motion_vectors = None
                    motion_magnitude = 0.0
                
                prev_frame = processed_frame.copy()
                
                # Anomaly detection
                anomaly_detected = self.anomaly_detector.detect_anomaly(
                    density_map, motion_magnitude
                )
                
                # Risk classification
                risk_level = self.risk_classifier.classify_risk(
                    density_value, motion_magnitude, anomaly_detected
                )
                
                # Calculate FPS
                current_fps = self.calculate_fps()
                
                # Update dashboard metrics
                self.dashboard.update_metrics(
                    density=density_value,
                    risk_level=risk_level,
                    motion_magnitude=motion_magnitude,
                    anomaly_detected=anomaly_detected,
                    fps=current_fps
                )
                
                # Generate alerts
                self.generate_alerts(
                    density_value, 
                    risk_level, 
                    anomaly_detected,
                    motion_magnitude
                )
                
                # Create visualization on original frame
                vis_frame = self.visualizer.draw_density_map(frame, density_map)
                
                if motion_vectors is not None:
                    vis_frame = self.visualizer.draw_motion_vectors(
                        vis_frame, motion_vectors
                    )
                
                vis_frame = self.visualizer.draw_risk_indicator(
                    vis_frame, risk_level
                )
                
                if anomaly_detected:
                    vis_frame = self.visualizer.draw_anomaly_alert(vis_frame)
                
                # Add frame info to visualization
                info_text = f"Frame: {frame_num}/{total_frames} | " \
                           f"Density: {density_value:.2f} | " \
                           f"Risk: {risk_level:.2f} | " \
                           f"Motion: {motion_magnitude:.2f}"
                
                cv2.putText(vis_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Render complete dashboard with visualization
                dashboard_frame = self.dashboard.render_dashboard(vis_frame)
                
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
                          f"Risk: {risk_level:.2f}", end='\r')
        
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 50)
            print("PROCESSING COMPLETE")
            print(f"Total frames processed: {frame_num}")
            print(f"Average FPS: {self.dashboard.fps_history and np.mean(self.dashboard.fps_history):.2f}")
            print(f"High risk frames: {self.dashboard.high_risk_frames}")
            print(f"Anomalies detected: {self.dashboard.anomaly_count}")
            print("=" * 50)


def main():
    """Main execution function"""
    # Initialize system
    system = EnhancedCrowdSafetySystem()
    
    # Input video path
    video_path = config.VIDEO_PATH
    
    # Output path
    output_dir = Path(config.OUTPUT_DIR)
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