import cv2
import numpy as np

class DashboardVisualizer:
    def __init__(self):
        self.frame_count = 0
    
    def create_pro_dashboard(self, frame, density, motion, risk, fps, 
                           density_history, risk_history, model_accuracy=None,
                           anomaly_score=None):
        """60% Video | 40% Pro Metrics + Graphs

        model_accuracy is accepted for backwards compatibility but is no
        longer drawn: this pipeline does not measure accuracy.
        """
        h, w = frame.shape[:2]
        
        # LEFT: Enhanced video (60%)
        video_w = int(w * 0.6)
        video_panel = cv2.resize(frame, (video_w, h))
        
        # PRO RISK OVERLAY
        if risk > 0.7:
            risk_text = "HIGH"
            risk_emoji = "!"
            risk_color = (0, 0, 255)
        elif risk > 0.4:
            risk_text = "MEDIUM"
            risk_emoji = "!"
            risk_color = (0, 255, 255)
        else:
            risk_text = "LOW"
            risk_emoji = ""
            risk_color = (0, 255, 0)
        
        cv2.rectangle(video_panel, (10,10), (video_w-10, 120), (20,20,20), -1)
        cv2.putText(video_panel, f"{risk_text} {risk_emoji} | Risk idx: {risk:.2f}", 
                   (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, risk_color, 3)
        cv2.putText(video_panel, "Crowd Safety AI", 
                   (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # RIGHT: PRO METRICS PANEL (40%)
        metrics_w = w - video_w
        metrics = np.zeros((h, metrics_w, 3), dtype=np.uint8)
        metrics.fill(15)
        
        # HEADER
        cv2.rectangle(metrics, (0,0), (metrics_w, 100), (50,50,50), -1)
        cv2.putText(metrics, "LIVE METRICS", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # KEY METRICS (FIXED - No question marks)
        metrics_data = [
            ("Density:", f"{density:.2f}", (0,255,0)),
            ("Motion:", f"{motion:.1f}px/f", (255,165,0)),
            ("Risk idx:", f"{risk:.2f}", (0,255,255)),
            ("Anomaly:", "n/a" if anomaly_score is None else f"{anomaly_score:.2f}", (255,0,255)),
            ("FPS:", f"{fps:.1f}", (0,255,0)),
            ("Frame:", f"{self.frame_count}", (255,255,255))
        ]
        
        y_pos = 140
        for label, value, color in metrics_data:
            cv2.putText(metrics, label, (25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(metrics, value, (metrics_w-90, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y_pos += 35
        
        # DUAL GRAPHS (Density + Risk trends) - Higher position
        graph_h = 140
        graph_y = 360
        
        # DENSITY GRAPH
        if len(density_history) > 20:
            x = np.linspace(20, metrics_w-20, len(density_history[-50:]))
            y = graph_y + 100 - np.array(density_history[-50:]) * 100
            points = np.column_stack((x, y)).astype(int)
            cv2.polylines(metrics, [points], False, (0,255,100), 2)
            cv2.putText(metrics, "Density Trend", (25, graph_y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,100), 1)
        
        # RISK GRAPH  
        graph_y2 = graph_y + 140
        if len(risk_history) > 20:
            x = np.linspace(20, metrics_w-20, len(risk_history[-50:]))
            y = graph_y2 + 100 - np.array(risk_history[-50:]) * 100
            points = np.column_stack((x, y)).astype(int)
            cv2.polylines(metrics, [points], False, (255,100,100), 2)
            cv2.putText(metrics, "Risk Trend", (25, graph_y2-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,100,100), 1)
        
        # THRESHOLD LINES for Risk Graph
        cv2.line(metrics, (25, graph_y2+70), (metrics_w-25, graph_y2+70), (255,255,0), 1)  # LOW
        cv2.line(metrics, (25, graph_y2+40), (metrics_w-25, graph_y2+40), (0,255,255), 1)  # MEDIUM
        cv2.line(metrics, (25, graph_y2+10), (metrics_w-25, graph_y2+10), (0,0,255), 1)   # HIGH
        
        # Footer note (replaces the former mock "accuracy" box)
        cv2.putText(metrics, "Unvalidated proxy metrics", (25, h-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

        # COMBINE
        dashboard = np.hstack([video_panel, metrics])
        self.frame_count += 1
        return dashboard
    
    def show_fullscreen(self, dashboard):
        cv2.namedWindow('Crowd Safety AI v2.0', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Crowd Safety AI v2.0', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Crowd Safety AI v2.0', dashboard)