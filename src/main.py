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

        # Full-length, per-frame records (the *_history lists above are
        # trimmed to the last 150 frames for the rolling in-video charts).
        self.records = []
        # Structured events: consecutive frames in the same state are merged
        # into one episode with a start/end frame and a peak value.
        self.events = []
        self._open_events = {}
        self.video_info = {}

    def _track_event(self, key, active, frame_num, t_sec, category, severity,
                     message, value):
        """Open, extend or close an event episode (frame-accurate)."""
        ev = self._open_events.get(key)
        if active:
            if ev is None:
                ev = {
                    'category': category, 'severity': severity,
                    'message': message,
                    'start_frame': frame_num, 'end_frame': frame_num,
                    'start_sec': round(t_sec, 3), 'end_sec': round(t_sec, 3),
                    'peak_value': float(value), 'frames': 1,
                }
                self._open_events[key] = ev
                self.events.append(ev)
            else:
                ev['end_frame'] = frame_num
                ev['end_sec'] = round(t_sec, 3)
                ev['frames'] += 1
                ev['peak_value'] = max(ev['peak_value'], float(value))
        elif ev is not None:
            del self._open_events[key]

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
    
    def process_video(self, video_path, output_path=None, display=True,
                      max_frames=None, progress_callback=None,
                      should_cancel=None):
        """Process video with enhanced dashboard visualization.

        Optional arguments (all backwards compatible):
          max_frames        stop after this many processed frames.
          progress_callback callable(frame_num, total_frames, info_dict)
                            called after every frame with real progress.
          should_cancel     callable() -> bool, checked once per frame.

        Returns a summary dict, or None if the video could not be opened.
        """
        # Load video - initialize VideoLoader with path and optimized settings
        try:
            # Use higher resolution and no frame skip for smoother playback
            video_loader = VideoLoader(video_path, resize_width=960, resize_height=540, frame_skip=1)
        except Exception as e:
            print(f"Error: Could not load video from {video_path}")
            print(f"Details: {e}")
            return
        
        # Get video properties from the VideoLoader's cap
        cap = video_loader.cap
        fps_original = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
        if fps_original <= 0 or fps_original > 240:
            fps_original = 25  # container did not report a usable FPS
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 0:
            total_frames = 0
        planned_frames = min(total_frames, max_frames) if (
            max_frames and total_frames) else (max_frames or total_frames)
        self.video_info = {
            'width': frame_width, 'height': frame_height,
            'fps': fps_original, 'total_frames': total_frames,
            'duration_sec': (total_frames / fps_original) if total_frames else 0.0,
        }
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps_original}")
        print(f"Total frames: {total_frames}")
        print("-" * 50)
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Dashboard output is 1920x1080
            out = cv2.VideoWriter(output_path, fourcc, fps_original, (1920, 1080))
            if not out.isOpened():
                print(f"Error: could not open video writer for {output_path}")
                out = None

        frame_num = 0
        prev_gray = None
        cancelled = False
        started = time.time()
        # NOTE: the previous hard-coded `model_accuracy = 92.5` placeholder
        # was removed. No accuracy is measured by this pipeline.
        model_accuracy = None
        
        try:
            while True:
                if should_cancel is not None and should_cancel():
                    cancelled = True
                    break
                if max_frames and frame_num >= max_frames:
                    break
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

                t_sec = frame_num / fps_original
                self.records.append({
                    'frame': frame_num,
                    't_sec': round(t_sec, 4),
                    'density': float(density_value),
                    'motion': float(motion_magnitude),
                    'risk_raw': float(risk_score),
                    'risk': float(risk_normalized),
                    'risk_label': risk_str,
                    'anomaly_score': float(anomaly_score),
                    'anomaly': bool(anomaly_detected),
                })
                self._track_event('risk_high', risk_normalized > 0.7,
                                  frame_num, t_sec, 'High risk index', 'high',
                                  'Risk index above 0.70', risk_normalized)
                self._track_event('anomaly', bool(anomaly_detected),
                                  frame_num, t_sec, 'Statistical anomaly',
                                  'medium',
                                  'Deviation from running mean above 0.70',
                                  anomaly_score)
                self._track_event('motion_high', motion_magnitude > 15.0,
                                  frame_num, t_sec, 'High motion', 'medium',
                                  'Mean optical-flow magnitude above 15 px/frame',
                                  motion_magnitude)
                self._track_event('density_high', density_value > 0.8,
                                  frame_num, t_sec, 'High density proxy',
                                  'high', 'Density proxy above 0.80',
                                  density_value)
                
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
                    model_accuracy,
                    anomaly_score=anomaly_score,
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
                        key = cv2.waitKey(1) & 0xFF
                
                # Write to output
                if out is not None:
                    out.write(dashboard_frame)
                
                if progress_callback is not None:
                    progress_callback(frame_num, planned_frames, {
                        'density': float(density_value),
                        'motion': float(motion_magnitude),
                        'risk': float(risk_normalized),
                        'risk_label': risk_str,
                        'anomaly_score': float(anomaly_score),
                    })

                # Progress indicator
                if frame_num % 30 == 0 and total_frames:
                    progress = (frame_num / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | "
                          f"FPS: {current_fps:.1f} | "
                          f"Risk: {risk_str} ({risk_normalized:.2f})", end='\r')
        
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            video_loader.release()
            if out is not None:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            print("\n" + "=" * 50)
            print("PROCESSING COMPLETE")
            print(f"Total frames processed: {frame_num}")
            print(f"Average FPS: {np.mean(self.dashboard.fps_history) if self.dashboard.fps_history else 0:.2f}")
            print(f"High risk frames: {self.dashboard.high_risk_frames}")
            print(f"Anomalies detected: {self.dashboard.anomaly_count}")
            print("=" * 50)

        self.summary = {
            'frames_processed': frame_num,
            'cancelled': cancelled,
            'elapsed_sec': round(time.time() - started, 3),
            'high_risk_frames': int(self.dashboard.high_risk_frames),
            'anomaly_frames': int(self.dashboard.anomaly_count),
            'video': dict(self.video_info),
            'output_written': out is not None,
        }
        return self.summary


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