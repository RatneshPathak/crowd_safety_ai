import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
import time

class CrowdSafetyDashboard:
    def __init__(self, max_history=100):
        self.max_history = max_history
        
        # Historical data storage
        self.density_history = deque(maxlen=max_history)
        self.risk_history = deque(maxlen=max_history)
        self.motion_history = deque(maxlen=max_history)
        self.anomaly_history = deque(maxlen=max_history)
        self.fps_history = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        
        # Statistics
        self.total_frames = 0
        self.high_risk_frames = 0
        self.anomaly_count = 0
        self.start_time = time.time()
        
        # Alert tracking
        self.current_alerts = []
        self.alert_history = deque(maxlen=10)
        
    def update_metrics(self, density, risk_level, motion_magnitude, anomaly_detected, fps):
        """Update all metrics with new data"""
        self.total_frames += 1
        current_time = time.time() - self.start_time
        
        # Update histories
        self.density_history.append(density)
        self.risk_history.append(risk_level)
        self.motion_history.append(motion_magnitude)
        self.anomaly_history.append(1 if anomaly_detected else 0)
        self.fps_history.append(fps)
        self.timestamps.append(current_time)
        
        # Update statistics
        if risk_level > 0.7:
            self.high_risk_frames += 1
        if anomaly_detected:
            self.anomaly_count += 1
            
    def add_alert(self, alert_type, severity, message):
        """Add a new alert"""
        alert = {
            'type': alert_type,
            'severity': severity,
            'message': message,
            'time': time.time() - self.start_time
        }
        self.current_alerts.append(alert)
        self.alert_history.append(alert)
        
    def clear_old_alerts(self, max_age=5.0):
        """Clear alerts older than max_age seconds"""
        current_time = time.time() - self.start_time
        self.current_alerts = [a for a in self.current_alerts 
                              if current_time - a['time'] < max_age]
    
    def create_line_chart(self, data, title, color, ylabel, width=400, height=200):
        """Create a line chart as numpy array"""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        if len(data) > 0:
            ax.plot(list(data), color=color, linewidth=2, alpha=0.8)
            ax.fill_between(range(len(data)), list(data), alpha=0.3, color=color)
        
        ax.set_title(title, color='white', fontsize=10, pad=5)
        ax.set_ylabel(ylabel, color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        ax.grid(True, alpha=0.2, color='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        
        plt.tight_layout()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        chart_img = np.asarray(buf)
        plt.close(fig)
        
        return cv2.cvtColor(chart_img, cv2.COLOR_RGBA2BGR)
    
    def create_gauge(self, value, title, max_value=1.0, width=250, height=180):
        """Create a gauge visualization"""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100, 
                              subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        # Gauge settings
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.plot(theta, [1]*len(theta), color='#3a3a3a', linewidth=20)
        
        # Value arc (color based on risk)
        value_normalized = min(value / max_value, 1.0)
        theta_value = theta[:int(len(theta) * value_normalized)]
        
        if value_normalized < 0.3:
            color = '#00ff00'
        elif value_normalized < 0.7:
            color = '#ffaa00'
        else:
            color = '#ff0000'
            
        ax.plot(theta_value, [1]*len(theta_value), color=color, linewidth=20)
        
        # Text
        ax.text(np.pi/2, 0.5, f'{value:.2f}', ha='center', va='center',
                fontsize=24, color='white', weight='bold')
        ax.text(np.pi/2, -0.3, title, ha='center', va='center',
                fontsize=10, color='white')
        
        ax.set_ylim(0, 1.2)
        ax.axis('off')
        plt.tight_layout()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        gauge_img = np.asarray(buf)
        plt.close(fig)
        
        return cv2.cvtColor(gauge_img, cv2.COLOR_RGBA2BGR)
    
    def create_stat_panel(self, width=400, height=300):
        """Create statistics panel"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (26, 26, 26)  # Dark background
        
        # Title
        cv2.putText(panel, 'SYSTEM STATISTICS', (20, 35),
                   cv2.FONT_HERSHEY_BOLD, 0.7, (255, 255, 255), 2)
        
        # Statistics
        uptime = time.time() - self.start_time
        avg_fps = np.mean(self.fps_history) if len(self.fps_history) > 0 else 0
        risk_percentage = (self.high_risk_frames / max(self.total_frames, 1)) * 100
        
        stats = [
            ('Uptime:', f'{int(uptime//60)}m {int(uptime%60)}s'),
            ('Total Frames:', f'{self.total_frames}'),
            ('Avg FPS:', f'{avg_fps:.1f}'),
            ('High Risk %:', f'{risk_percentage:.1f}%'),
            ('Anomalies:', f'{self.anomaly_count}'),
        ]
        
        y_pos = 80
        for label, value in stats:
            cv2.putText(panel, label, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(panel, value, (220, y_pos),
                       cv2.FONT_HERSHEY_BOLD, 0.7, (0, 255, 0), 2)
            y_pos += 40
            
        return panel
    
    def create_alert_panel(self, width=400, height=250):
        """Create alert/notification panel"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (26, 26, 26)
        
        # Title
        cv2.putText(panel, 'ACTIVE ALERTS', (20, 35),
                   cv2.FONT_HERSHEY_BOLD, 0.7, (255, 255, 255), 2)
        
        # Draw alerts
        y_pos = 75
        if not self.current_alerts:
            cv2.putText(panel, 'No active alerts', (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1)
        else:
            for alert in self.current_alerts[-5:]:  # Show last 5 alerts
                color = {'high': (0, 0, 255), 'medium': (0, 165, 255), 
                        'low': (0, 255, 255)}.get(alert['severity'], (255, 255, 255))
                
                # Alert icon
                cv2.circle(panel, (25, y_pos-5), 5, color, -1)
                
                # Alert text
                cv2.putText(panel, alert['message'][:40], (40, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 35
                
        return panel
    
    def create_heatmap_legend(self, width=200, height=150):
        """Create a heatmap color legend"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (26, 26, 26)
        
        cv2.putText(panel, 'DENSITY MAP', (20, 30),
                   cv2.FONT_HERSHEY_BOLD, 0.6, (255, 255, 255), 2)
        
        # Color gradient
        gradient = np.zeros((80, 160, 3), dtype=np.uint8)
        for i in range(160):
            value = int((i / 160) * 255)
            color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), 
                                     cv2.COLORMAP_JET)[0][0]
            gradient[:, i] = color
            
        panel[50:130, 20:180] = gradient
        
        # Labels
        cv2.putText(panel, 'Low', (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        cv2.putText(panel, 'High', (140, 145), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        return panel
    
    def render_dashboard(self, main_frame):
        """Render complete dashboard with main video frame"""
        # Get current metrics
        current_density = self.density_history[-1] if self.density_history else 0
        current_risk = self.risk_history[-1] if self.risk_history else 0
        current_motion = self.motion_history[-1] if self.motion_history else 0
        current_fps = self.fps_history[-1] if self.fps_history else 0
        
        # Create dashboard components
        density_chart = self.create_line_chart(self.density_history, 
            'Crowd Density Over Time', '#00ffff', 'Density')
        risk_chart = self.create_line_chart(self.risk_history,
            'Risk Level Over Time', '#ff0000', 'Risk')
        motion_chart = self.create_line_chart(self.motion_history,
            'Motion Intensity', '#00ff00', 'Motion')
        
        risk_gauge = self.create_gauge(current_risk, 'Risk Level')
        density_gauge = self.create_gauge(current_density, 'Density')
        
        stat_panel = self.create_stat_panel()
        alert_panel = self.create_alert_panel()
        heatmap_legend = self.create_heatmap_legend()
        
        # Layout configuration
        main_h, main_w = main_frame.shape[:2]
        
        # Create dashboard layout (1920x1080 standard)
        dashboard_w = 1920
        dashboard_h = 1080
        dashboard = np.zeros((dashboard_h, dashboard_w, 3), dtype=np.uint8)
        dashboard[:] = (20, 20, 20)
        
        # Resize main frame to fit (center-left)
        main_display_w = 1200
        main_display_h = 900
        main_resized = cv2.resize(main_frame, (main_display_w, main_display_h))
        
        # Place main video (with border)
        border_size = 3
        cv2.rectangle(dashboard, (20-border_size, 20-border_size),
                     (20+main_display_w+border_size, 20+main_display_h+border_size),
                     (100, 100, 255), border_size)
        dashboard[20:20+main_display_h, 20:20+main_display_w] = main_resized
        
        # Header
        cv2.rectangle(dashboard, (0, 0), (dashboard_w, 60), (40, 40, 40), -1)
        cv2.putText(dashboard, 'CROWD SAFETY MONITORING SYSTEM', (30, 40),
                   cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 3)
        
        # FPS Counter
        cv2.putText(dashboard, f'FPS: {current_fps:.1f}', (dashboard_w-150, 40),
                   cv2.FONT_HERSHEY_BOLD, 0.8, (0, 255, 0), 2)
        
        # Right panel - Charts and gauges
        right_x = 1240
        y_offset = 80
        
        # Gauges
        dashboard[y_offset:y_offset+180, right_x:right_x+250] = risk_gauge
        dashboard[y_offset:y_offset+180, right_x+270:right_x+520] = density_gauge
        y_offset += 200
        
        # Charts
        dashboard[y_offset:y_offset+200, right_x:right_x+400] = density_chart
        y_offset += 220
        
        dashboard[y_offset:y_offset+200, right_x:right_x+400] = risk_chart
        y_offset += 220
        
        dashboard[y_offset:y_offset+200, right_x:right_x+400] = motion_chart
        
        # Bottom panels
        bottom_y = 940
        dashboard[bottom_y:bottom_y+300, 20:420] = stat_panel
        dashboard[bottom_y:bottom_y+250, 440:840] = alert_panel
        dashboard[bottom_y:bottom_y+150, 860:1060] = heatmap_legend
        
        # Status indicator
        status_color = (0, 255, 0) if current_risk < 0.5 else \
                      (0, 165, 255) if current_risk < 0.7 else (0, 0, 255)
        status_text = 'SAFE' if current_risk < 0.5 else \
                     'CAUTION' if current_risk < 0.7 else 'DANGER'
        
        cv2.circle(dashboard, (dashboard_w-100, 35), 15, status_color, -1)
        cv2.putText(dashboard, status_text, (dashboard_w-200, 85),
                   cv2.FONT_HERSHEY_BOLD, 0.6, status_color, 2)
        
        return dashboard


# Integration example for your main.py
def example_integration():
    """
    Add this to your existing main.py processing loop:
    
    # Initialize dashboard
    dashboard = CrowdSafetyDashboard(max_history=100)
    
    # In your processing loop:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Your existing processing
        density = calculate_density(frame)
        risk = classify_risk(frame)
        motion = analyze_motion(frame)
        anomaly = detect_anomaly(frame)
        
        # Update dashboard
        dashboard.update_metrics(density, risk, motion, anomaly, fps)
        
        # Add alerts based on conditions
        if risk > 0.8:
            dashboard.add_alert('High Risk', 'high', 
                              'Critical crowd density detected!')
        if anomaly:
            dashboard.add_alert('Anomaly', 'medium', 
                              'Unusual movement pattern detected')
        
        # Clear old alerts
        dashboard.clear_old_alerts(max_age=5.0)
        
        # Render final output
        output_frame = dashboard.render_dashboard(processed_frame)
        
        # Display or save
        cv2.imshow('Crowd Safety Dashboard', output_frame)
        out.write(output_frame)
    """
    pass