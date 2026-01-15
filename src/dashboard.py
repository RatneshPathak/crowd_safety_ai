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
    
    def create_line_chart(self, data, title, color, ylabel, width=450, height=220):
        """Create a line chart as numpy array"""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        
        if len(data) > 0:
            ax.plot(list(data), color=color, linewidth=2.5, alpha=0.9)
            ax.fill_between(range(len(data)), list(data), alpha=0.4, color=color)
        
        ax.set_title(title, color='white', fontsize=11, pad=8, weight='bold')
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.3, color='white', linestyle='--')
        ax.spines['bottom'].set_color('#666666')
        ax.spines['top'].set_color('#666666')
        ax.spines['right'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        
        plt.tight_layout()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        chart_img = np.asarray(buf)
        plt.close(fig)
        
        return cv2.cvtColor(chart_img, cv2.COLOR_RGBA2BGR)
    
    def create_gauge(self, value, title, max_value=1.0, width=280, height=200):
        """Create a gauge visualization"""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100, 
                              subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        # Gauge settings
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.plot(theta, [1]*len(theta), color='#3a3a3a', linewidth=22)
        
        # Value arc (color based on risk)
        value_normalized = min(value / max_value, 1.0)
        theta_value = theta[:int(len(theta) * value_normalized)]
        
        if value_normalized < 0.3:
            color = '#00ff00'
        elif value_normalized < 0.7:
            color = '#ffaa00'
        else:
            color = '#ff0000'
            
        ax.plot(theta_value, [1]*len(theta_value), color=color, linewidth=22)
        
        # Main value text
        ax.text(np.pi/2, 0.5, f'{value:.2f}', ha='center', va='center',
                fontsize=28, color='white', weight='bold')
        
        # Title text (larger, bolder)
        ax.text(np.pi/2, -0.25, title, ha='center', va='center',
                fontsize=12, color='white', weight='bold')
        
        ax.set_ylim(0, 1.2)
        ax.axis('off')
        plt.tight_layout()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        gauge_img = np.asarray(buf)
        plt.close(fig)
        
        return cv2.cvtColor(gauge_img, cv2.COLOR_RGBA2BGR)
    
    def create_stat_panel(self, width=420, height=140):
        """Create statistics panel"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (26, 26, 26)  # Dark background
        
        # Title with background
        cv2.rectangle(panel, (0, 0), (width, 35), (40, 40, 40), -1)
        cv2.putText(panel, 'SYSTEM STATISTICS', (15, 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        # Statistics
        uptime = time.time() - self.start_time
        avg_fps = np.mean(self.fps_history) if len(self.fps_history) > 0 else 0
        risk_percentage = (self.high_risk_frames / max(self.total_frames, 1)) * 100
        
        stats = [
            ('Uptime:', f'{int(uptime//60)}m {int(uptime%60)}s', (100, 200, 255)),
            ('Frames:', f'{self.total_frames}', (100, 255, 100)),
            ('Avg FPS:', f'{avg_fps:.1f}', (255, 200, 100)),
            ('Risk:', f'{risk_percentage:.1f}%', (255, 100, 100)),
        ]
        
        y_pos = 62
        for label, value, color in stats:
            cv2.putText(panel, label, (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(panel, value, (190, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y_pos += 28
            
        return panel
    
    def create_alert_panel(self, width=520, height=140):
        """Create alert/notification panel"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (26, 26, 26)
        
        # Title with background
        cv2.rectangle(panel, (0, 0), (width, 35), (40, 40, 40), -1)
        cv2.putText(panel, 'ACTIVE ALERTS', (15, 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 100), 2)
        
        # Draw alerts
        y_pos = 62
        if not self.current_alerts:
            cv2.putText(panel, 'All systems normal - No alerts', (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        else:
            for alert in self.current_alerts[-3:]:  # Show last 3 alerts
                color = {'high': (0, 0, 255), 'medium': (0, 165, 255), 
                        'low': (0, 255, 255)}.get(alert['severity'], (255, 255, 255))
                
                # Alert icon
                cv2.circle(panel, (20, y_pos-6), 6, color, -1)
                
                # Alert text
                text = alert['message'][:55]
                cv2.putText(panel, text, (35, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y_pos += 28
                
        return panel
    
    def create_heatmap_legend(self, width=280, height=140):
        """Create a heatmap color legend"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (26, 26, 26)
        
        # Title with background
        cv2.rectangle(panel, (0, 0), (width, 35), (40, 40, 40), -1)
        cv2.putText(panel, 'DENSITY SCALE', (15, 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 2)
        
        # Color gradient
        gradient = np.zeros((65, 240, 3), dtype=np.uint8)
        for i in range(240):
            value = int((i / 240) * 255)
            color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), 
                                     cv2.COLORMAP_JET)[0][0]
            gradient[:, i] = color
            
        panel[48:113, 20:260] = gradient
        
        # Labels
        cv2.putText(panel, 'Low', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.45, (255, 255, 255), 1)
        cv2.putText(panel, 'Medium', (110, 130), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.45, (255, 255, 255), 1)
        cv2.putText(panel, 'High', (220, 130), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.45, (255, 255, 255), 1)
        
        return panel
    
    def render_dashboard(self, main_frame, model_accuracy=92.5):
        """Render complete dashboard with main video frame"""
        # Get current metrics
        current_density = self.density_history[-1] if self.density_history else 0
        current_risk = self.risk_history[-1] if self.risk_history else 0
        current_motion = self.motion_history[-1] if self.motion_history else 0
        current_fps = self.fps_history[-1] if self.fps_history else 0
        
        # Create dashboard components with improved sizes
        density_chart = self.create_line_chart(self.density_history, 
            'Crowd Density Trend', '#00ffff', 'Density', width=450, height=220)
        risk_chart = self.create_line_chart(self.risk_history,
            'Risk Level Trend', '#ff4444', 'Risk', width=450, height=220)
        motion_chart = self.create_line_chart(self.motion_history,
            'Motion Activity', '#44ff44', 'Motion', width=450, height=220)
        
        risk_gauge = self.create_gauge(current_risk, 'RISK LEVEL', width=280, height=200)
        density_gauge = self.create_gauge(current_density, 'DENSITY', width=280, height=200)
        
        stat_panel = self.create_stat_panel(width=420, height=140)
        alert_panel = self.create_alert_panel(width=520, height=140)
        heatmap_legend = self.create_heatmap_legend(width=280, height=140)
        
        # Layout configuration (1920x1080)
        dashboard_w = 1920
        dashboard_h = 1080
        dashboard = np.zeros((dashboard_h, dashboard_w, 3), dtype=np.uint8)
        dashboard[:] = (20, 20, 20)
        
        # Resize main frame to fit (larger display)
        main_display_w = 1180
        main_display_h = 885
        main_resized = cv2.resize(main_frame, (main_display_w, main_display_h))
        
        # Place main video (with stylish border)
        border_size = 4
        cv2.rectangle(dashboard, (20-border_size, 75-border_size),
                     (20+main_display_w+border_size, 75+main_display_h+border_size),
                     (0, 200, 255), border_size)
        dashboard[75:75+main_display_h, 20:20+main_display_w] = main_resized
        
        # Header with gradient effect
        for i in range(60):
            alpha = 1.0 - (i / 60.0) * 0.3
            color_val = int(40 * alpha)
            cv2.line(dashboard, (0, i), (dashboard_w, i), (color_val, color_val, color_val), 1)
        
        cv2.putText(dashboard, 'CROWD SAFETY MONITORING SYSTEM', (35, 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        
        # FPS Counter with background
        fps_w = 180
        cv2.rectangle(dashboard, (dashboard_w-fps_w-20, 15), (dashboard_w-20, 55), 
                     (40, 60, 40), -1)
        cv2.putText(dashboard, f'FPS: {current_fps:.1f}', (dashboard_w-fps_w, 43),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Right panel - Charts and gauges (improved layout with better spacing)
        right_x = 1220
        y_offset = 75
        
        # Gauges side by side
        dashboard[y_offset:y_offset+200, right_x:right_x+280] = risk_gauge
        dashboard[y_offset:y_offset+200, right_x+300:right_x+580] = density_gauge
        y_offset += 215
        
        # Charts stacked with proper spacing
        dashboard[y_offset:y_offset+220, right_x:right_x+450] = density_chart
        y_offset += 230
        
        dashboard[y_offset:y_offset+220, right_x:right_x+450] = risk_chart
        y_offset += 230
        
        dashboard[y_offset:y_offset+220, right_x:right_x+450] = motion_chart
        
        # Bottom panels (better spacing)
        bottom_y = 940
        dashboard[bottom_y:bottom_y+140, 20:440] = stat_panel
        dashboard[bottom_y:bottom_y+140, 460:980] = alert_panel
        dashboard[bottom_y:bottom_y+140, 1000:1280] = heatmap_legend
        
        # Status indicator (larger and more visible)
        status_color = (0, 255, 0) if current_risk < 0.5 else \
                      (0, 165, 255) if current_risk < 0.7 else (0, 0, 255)
        status_text = 'SAFE' if current_risk < 0.5 else \
                     'CAUTION' if current_risk < 0.7 else 'DANGER'
        
        # Status circle with glow effect
        for r in range(20, 15, -1):
            alpha = (20 - r) / 5.0
            glow_color = tuple(int(c * alpha) for c in status_color)
            cv2.circle(dashboard, (dashboard_w-90, 35), r, glow_color, 2)
        cv2.circle(dashboard, (dashboard_w-90, 35), 15, status_color, -1)
        
        # Status text with background
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(dashboard, (dashboard_w-250, 10), 
                     (dashboard_w-250 + text_size[0] + 20, 60), (30, 30, 30), -1)
        cv2.putText(dashboard, status_text, (dashboard_w-240, 43),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return dashboard