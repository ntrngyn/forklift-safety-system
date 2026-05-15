import json
import os
import numpy as np

class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        # Tọa độ mặc định
        self.src_pts = np.float32([[1.0, 167.0], [638.0, 164.0], [638.0, 354.0], [2.0, 354.0]])
        self.dst_pts = np.float32([[350, 200], [650, 200], [650, 800], [350, 800]])
        self.pixels_per_meter = 65.86
        self.danger_dist_m = 2.0
        self.warning_dist_m = 5.0
        
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.src_pts = np.float32(data['src_pts'])
                    self.pixels_per_meter = float(data.get('pixels_per_meter', 65.86))
                    self.danger_dist_m = float(data.get('danger_dist_m', 2.0))
                    self.warning_dist_m = float(data.get('warning_dist_m', 3.5))
                print("Đã tải cấu hình từ file config.json")
            except Exception as e:
                print(f"Lỗi đọc file cấu hình: {e}")

    def save_config(self):
        data = {
            'src_pts': self.src_pts.tolist(),
            'pixels_per_meter': self.pixels_per_meter,
            'danger_dist_m': self.danger_dist_m,     
            'warning_dist_m': self.warning_dist_m    
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=4)
        print("Đã lưu cấu hình mới vào config.json")