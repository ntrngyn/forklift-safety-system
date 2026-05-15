from ultralytics import YOLO
import os

class AIEngine:
    def __init__(self):
        self.model_path = 'weights/best.pt'
        self.model = None
        self.load_model(self.model_path)

    def load_model(self, path):
        try:
            if os.path.exists(path):
                self.model = YOLO(path)
                return True
            else:
                print(f"File model không tồn tại tại: {path}")
                return False
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            return False

    def detect(self, frame, conf_threshold):
        if self.model is None:
            return None
        return self.model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=conf_threshold, 
            verbose=False
        )