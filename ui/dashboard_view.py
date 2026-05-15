import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import numpy as np
import datetime
import time
import os
from tkinter import filedialog 

# Import các module cốt lõi
from core.config_manager import ConfigManager
from core.geometry_utils import GeometryUtils
from core.ai_engine import AIEngine

# ==========================================
# CẤU HÌNH GIAO DIỆN (UI)
# ==========================================
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue")

class WarehouseDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hệ Thống Giám Sát An Toàn Trí Tuệ Nhân Tạo")
        self.geometry("1366x768") 
        
        # --- KHỞI TẠO CÁC MODULE LÕI ---
        self.cfg_manager = ConfigManager()
        self.geo_utils = GeometryUtils(self.cfg_manager)
        self.ai = AIEngine()

        # --- QUẢN LÝ TRẠNG THÁI VIDEO & THÔNG SỐ ---
        self.source_type = None 
        self.video_path = None
        self.image_path = None
        self.conf_threshold = 0.259
        self.cap = None
        self.is_running = False
        self.last_log_time = 0 
        
        # --- QUẢN LÝ TRẠNG THÁI GIAO DIỆN ---
        self.radar_visible = True 
        self.camera_visible = True 
        
        self.setup_ui()

    def setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=320)
        self.grid_columnconfigure(1, weight=3) 
        self.grid_columnconfigure(2, weight=1) 

        self.bind("<Key>", self.handle_keypress)

        # ==========================================
        # 1. SIDEBAR (CỘT TRÁI)
        # ==========================================
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="CONTROL PANEL", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Trạng thái: ĐÃ TẮT", text_color="gray", font=ctk.CTkFont(size=16, weight="bold"), width=280)
        self.status_label.grid(row=1, column=0, padx=20, pady=5)

        self.btn_load_model = ctk.CTkButton(self.sidebar_frame, text="Chọn Model AI (.pt)", fg_color="#3a3a3a", hover_color="#4a4a4a", command=self.select_model)
        self.btn_load_model.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="ew")

        self.btn_start = ctk.CTkButton(self.sidebar_frame, text="Khởi động Camera", fg_color="#2fa572", hover_color="#25825a", command=self.start_camera)
        self.btn_start.grid(row=3, column=0, padx=20, pady=5, sticky="ew")

        self.btn_load_video = ctk.CTkButton(self.sidebar_frame, text="Tải Video Test", fg_color="#3a3a3a", hover_color="#4a4a4a", command=self.load_video)
        self.btn_load_video.grid(row=4, column=0, padx=20, pady=5, sticky="ew")

        self.btn_load_image = ctk.CTkButton(self.sidebar_frame, text="Tải Ảnh Test", fg_color="#3a3a3a", hover_color="#4a4a4a", command=self.load_image)
        self.btn_load_image.grid(row=5, column=0, padx=20, pady=5, sticky="ew")

        self.btn_stop = ctk.CTkButton(self.sidebar_frame, text="Dừng hệ thống", fg_color="#c93434", hover_color="#a32a2a", command=self.stop_camera)
        self.btn_stop.grid(row=6, column=0, padx=20, pady=(5, 15), sticky="ew")

        self.config_label = ctk.CTkLabel(self.sidebar_frame, text="-- CÔNG CỤ HIỆU CHUẨN --", font=ctk.CTkFont(size=12, weight="bold"))
        self.config_label.grid(row=7, column=0, pady=(5, 5))

        self.btn_calib_floor = ctk.CTkButton(self.sidebar_frame, text="1. Chọn 4 điểm mặt sàn", fg_color="#1f538d", hover_color="#14375e", command=self.calibrate_floor)
        self.btn_calib_floor.grid(row=8, column=0, padx=20, pady=5, sticky="ew")

        self.ref_dist_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.ref_dist_frame.grid(row=9, column=0, padx=20, pady=2, sticky="ew")
        self.ref_dist_frame.grid_columnconfigure(0, weight=1)
        self.ref_dist_frame.grid_columnconfigure(1, weight=1)

        self.lbl_ref_dist = ctk.CTkLabel(self.ref_dist_frame, text="Chiều dài thực (m):", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_ref_dist.grid(row=0, column=0, sticky="w")
        
        self.entry_ref_dist = ctk.CTkEntry(self.ref_dist_frame, width=60)
        self.entry_ref_dist.grid(row=0, column=1, sticky="e")
        self.entry_ref_dist.insert(0, "1.07") 

        self.btn_calib_scale = ctk.CTkButton(self.sidebar_frame, text="2. Cân chỉnh tỷ lệ", fg_color="#1f538d", hover_color="#14375e", command=self.calibrate_scale)
        self.btn_calib_scale.grid(row=10, column=0, padx=20, pady=(5, 10), sticky="ew")

        self.dist_label = ctk.CTkLabel(self.sidebar_frame, text="-- QUY ĐỊNH KHOẢNG CÁCH --", font=ctk.CTkFont(size=12, weight="bold"))
        self.dist_label.grid(row=11, column=0, pady=(5, 5)) 

        self.dist_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.dist_frame.grid(row=12, column=0, padx=20, pady=0, sticky="ew") 
        self.dist_frame.grid_columnconfigure(0, weight=1)
        self.dist_frame.grid_columnconfigure(1, weight=1)

        self.lbl_danger = ctk.CTkLabel(self.dist_frame, text="Nguy hiểm (Đỏ):", text_color="#ff4d4d", font=ctk.CTkFont(weight="bold"))
        self.lbl_danger.grid(row=0, column=0, sticky="w", pady=2)
        self.entry_danger = ctk.CTkEntry(self.dist_frame, width=60)
        self.entry_danger.grid(row=0, column=1, sticky="e", pady=2)
        self.entry_danger.insert(0, str(self.cfg_manager.danger_dist_m)) 

        self.lbl_warning = ctk.CTkLabel(self.dist_frame, text="Cảnh báo (Vàng):", text_color="#ffcc00", font=ctk.CTkFont(weight="bold"))
        self.lbl_warning.grid(row=1, column=0, sticky="w", pady=2)
        self.entry_warning = ctk.CTkEntry(self.dist_frame, width=60)
        self.entry_warning.grid(row=1, column=1, sticky="e", pady=2)
        self.entry_warning.insert(0, str(self.cfg_manager.warning_dist_m)) 

        self.btn_save_dist = ctk.CTkButton(self.sidebar_frame, text="3. Cập nhật khoảng cách", fg_color="#1f538d", hover_color="#14375e", command=self.update_distances)
        self.btn_save_dist.grid(row=13, column=0, padx=20, pady=5, sticky="ew") 

        self.sidebar_frame.grid_rowconfigure(15, weight=1) 
        
        self.log_label = ctk.CTkLabel(self.sidebar_frame, text="NHẬT KÝ CẢNH BÁO", font=ctk.CTkFont(size=14, weight="bold"))
        self.log_label.grid(row=14, column=0, sticky="s", pady=(5, 5)) 

        self.log_textbox = ctk.CTkTextbox(self.sidebar_frame)
        self.log_textbox.grid(row=15, column=0, padx=10, pady=(0, 20), sticky="nsew")

        # ==========================================
        # 2. CAMERA VIDEO (CỘT GIỮA)
        # ==========================================
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.video_frame.grid_rowconfigure(1, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_header = ctk.CTkFrame(self.video_frame, fg_color="transparent")
        self.video_header.grid(row=0, column=0, sticky="ew", pady=(10, 0), padx=10)
        self.video_header.grid_columnconfigure(0, weight=1)

        self.video_title = ctk.CTkLabel(self.video_header, text="CAMERA GIÁM SÁT KHO (Phím tắt: C)", font=ctk.CTkFont(size=14, weight="bold"))
        self.video_title.grid(row=0, column=0, sticky="w")

        self.btn_toggle_camera = ctk.CTkButton(self.video_header, text="◀", width=30, height=30, fg_color="#3a3a3a", hover_color="#4a4a4a", command=self.toggle_camera)
        self.btn_toggle_camera.grid(row=0, column=1, sticky="e")

        self.video_label = ctk.CTkLabel(self.video_frame, text="Màn hình chờ...")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # ==========================================
        # 3. BẢN ĐỒ RADAR (CỘT PHẢI)
        # ==========================================
        self.radar_frame = ctk.CTkFrame(self)
        self.radar_frame.grid(row=0, column=2, sticky="nsew", padx=(0, 10), pady=10)
        self.radar_frame.grid_rowconfigure(1, weight=1)
        self.radar_frame.grid_columnconfigure(0, weight=1)

        self.radar_header = ctk.CTkFrame(self.radar_frame, fg_color="transparent")
        self.radar_header.grid(row=0, column=0, sticky="ew", pady=(10, 0), padx=10)
        self.radar_header.grid_columnconfigure(0, weight=1)

        self.radar_title = ctk.CTkLabel(self.radar_header, text="BẢN ĐỒ RADAR 2D (Phím tắt: R)", font=ctk.CTkFont(size=14, weight="bold"))
        self.radar_title.grid(row=0, column=0, sticky="w")

        self.btn_toggle_radar = ctk.CTkButton(self.radar_header, text="▶", width=30, height=30, fg_color="#3a3a3a", hover_color="#4a4a4a", command=self.toggle_radar)
        self.btn_toggle_radar.grid(row=0, column=1, sticky="e")

        self.radar_label = ctk.CTkLabel(self.radar_frame, text="")
        self.radar_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    # ==========================================
    # LOGIC CẬP NHẬT TỶ LỆ GRID HOÀN HẢO
    # ==========================================
    def refresh_grid_layout(self):
        self.video_label.configure(image="")
        self.radar_label.configure(image="")
        
        if self.camera_visible and self.radar_visible:
            self.grid_columnconfigure(1, weight=3)
            self.grid_columnconfigure(2, weight=1)
        elif self.camera_visible and not self.radar_visible:
            self.grid_columnconfigure(1, weight=1)
            self.grid_columnconfigure(2, weight=0)
        elif not self.camera_visible and self.radar_visible:
            self.grid_columnconfigure(1, weight=0)
            self.grid_columnconfigure(2, weight=1)
        else:
            self.grid_columnconfigure(1, weight=0)
            self.grid_columnconfigure(2, weight=0)

    # ==========================================
    # LOGIC ĐIỀU KHIỂN & SỰ KIỆN GIAO DIỆN
    # ==========================================
    def handle_keypress(self, event):
        widget = self.focus_get()
        if isinstance(widget, ctk.CTkEntry) or isinstance(widget, ctk.CTkTextbox):
            return 
            
        char = event.char.lower() if event.char else ""
        if char == 'c':
            self.toggle_camera()
        elif char == 'r':
            self.toggle_radar()

    def toggle_camera(self):
        if self.camera_visible:
            self.video_label.grid_remove() 
            self.video_title.configure(text="") 
            self.btn_toggle_camera.configure(text="▶") 
            self.camera_visible = False
            self.log_alert("ℹ️ Đã thu gọn Camera để ưu tiên hiển thị Radar.")
        else:
            self.video_label.grid()
            self.video_title.configure(text="CAMERA GIÁM SÁT KHO (Phím tắt: C)")
            self.btn_toggle_camera.configure(text="◀")
            self.camera_visible = True
            self.log_alert("ℹ️ Đã mở rộng lại màn hình Camera.")
        self.refresh_grid_layout()

    def toggle_radar(self):
        if self.radar_visible:
            self.radar_label.grid_remove() 
            self.radar_title.configure(text="") 
            self.btn_toggle_radar.configure(text="◀")
            self.radar_visible = False
            self.log_alert("ℹ️ Đã thu gọn bản đồ Radar.")
        else:
            self.radar_label.grid()
            self.radar_title.configure(text="BẢN ĐỒ RADAR 2D (Phím tắt: R)")
            self.btn_toggle_radar.configure(text="▶")
            self.radar_visible = True
            self.log_alert("ℹ️ Đã mở rộng bản đồ Radar.")
        self.refresh_grid_layout()

    def select_model(self):
        file_path = filedialog.askopenfilename(title="Chọn Model YOLO", filetypes=[("YOLO Weights", "*.pt")])
        if file_path:
            was_running = self.is_running
            self.stop_camera() 
            success = self.ai.load_model(file_path)
            if success:
                model_name = os.path.basename(file_path)
                self.log_alert(f"✅ Đã tải model mới: {model_name}")
                if was_running:
                    if self.source_type == 'video': self.load_video()
                    elif self.source_type == 'camera': self.start_camera()
                    elif self.source_type == 'image': self.load_image()
            else:
                self.log_alert(f"❌ Lỗi tải model!")

    def start_camera(self):
        self.stop_camera()
        self.cap = cv2.VideoCapture(0) 
        self.source_type = 'camera'
        self.is_running = True
        self.status_label.configure(text="Trạng thái: CAMERA", text_color="green")
        self.update_video()

    def load_video(self):
        file_path = filedialog.askopenfilename(title="Chọn Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.stop_camera()
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            self.source_type = 'video'
            self.is_running = True
            self.status_label.configure(text="Trạng thái: PHÁT VIDEO", text_color="#2b719e")
            self.update_video()

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Chọn Ảnh", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.stop_camera()
            self.image_path = file_path
            self.source_type = 'image'
            self.is_running = True
            self.status_label.configure(text="Trạng thái: ẢNH TĨNH", text_color="#8b4513")
            self.update_video()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.source_type = None
        self.video_label.configure(image="", text="Màn hình chờ...")
        self.radar_label.configure(image="")
        self.status_label.configure(text="Trạng thái: ĐÃ TẮT", text_color="gray")

    def log_alert(self, msg):
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_textbox.insert("0.0", f"[{time_str}] {msg}\n")
        self.log_textbox.see("end")

    def update_distances(self):
        try:
            d_val = float(self.entry_danger.get())
            w_val = float(self.entry_warning.get())
            if d_val >= w_val:
                self.log_alert("❌ Lỗi: Khoảng cách Đỏ phải nhỏ hơn Vàng!")
                return
            if d_val <= 0 or w_val <= 0:
                self.log_alert("❌ Lỗi: Khoảng cách phải lớn hơn 0!")
                return
            
            # Ghi vào ConfigManager thay vì lưu trực tiếp ở UI
            self.cfg_manager.danger_dist_m = d_val
            self.cfg_manager.warning_dist_m = w_val
            self.cfg_manager.save_config()
            self.log_alert(f"✅ Đã lưu KC: Nguy hiểm ({d_val}m), Cảnh báo ({w_val}m)")
        except ValueError:
            self.log_alert("❌ Lỗi: Vui lòng nhập số hợp lệ (VD: 2.0)!")

    # ==========================================
    # CÔNG CỤ HIỆU CHUẨN CAMERA
    # ==========================================
    def get_current_frame(self):
        if self.source_type == 'image' and self.image_path:
            return cv2.imread(self.image_path)
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        else:
            self.log_alert("ℹ️ Đang mở hộp thoại chọn ảnh để hiệu chuẩn...")
            file_path = filedialog.askopenfilename(
                title="Chọn Ảnh Hiệu Chuẩn (Mặt Sàn / Xe Nâng)", 
                filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
            )
            if file_path: return cv2.imread(file_path)
            else:
                self.log_alert("❌ Đã hủy chọn ảnh hiệu chuẩn.")
                return None

    def calibrate_floor(self):
        was_running = self.is_running
        self.stop_camera() 
        
        frame = self.get_current_frame()
        if frame is None:
            self.log_alert("❌ Lỗi: Không lấy được khung hình để hiệu chuẩn!")
            return

        points = []
        window_name = "Chon 4 diem theo thu tu (Trai-Tren > Phai-Tren > Phai-Duoi > Trai-Duoi)"
        
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                if len(points) == 4:
                    pts = np.array(points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.imshow(window_name, frame)

        cv2.imshow(window_name, frame)
        cv2.setMouseCallback(window_name, mouse_cb)
        
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == 27 or key == ord('q'): break 
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
            if len(points) == 4:
                cv2.waitKey(1000) 
                break
                
        cv2.destroyAllWindows()
        
        if len(points) == 4:
            # Truyền dữ liệu cho Config và Geometry
            self.cfg_manager.src_pts = np.float32(points)
            self.geo_utils.update_matrix() 
            self.cfg_manager.save_config() 
            self.log_alert("✅ Cập nhật mặt sàn và lưu thành công!")
        else:
            self.log_alert("❌ Đã hủy chọn mặt sàn.")
            
        if was_running: 
            if self.source_type == 'video': self.load_video()
            elif self.source_type == 'camera': self.start_camera()
            elif self.source_type == 'image': self.load_image()

    def calibrate_scale(self):
        try:
            ref_length = float(self.entry_ref_dist.get())
            if ref_length <= 0:
                self.log_alert("❌ Lỗi: Chiều dài thực tế phải lớn hơn 0!")
                return
        except ValueError:
            self.log_alert("❌ Lỗi: Vui lòng nhập số hợp lệ (VD: 1.07, 0.5)!")
            return

        was_running = self.is_running
        self.stop_camera() 
        
        frame = self.get_current_frame()
        if frame is None: 
            self.log_alert("❌ Lỗi: Không lấy được khung hình để hiệu chuẩn!")
            return

        points = []
        window_name = f"Chon 2 diem tren anh dai tuong duong {ref_length} met"
        
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append([x, y])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                if len(points) == 2:
                    cv2.line(frame, tuple(points[0]), tuple(points[1]), (255, 0, 0), 2)
                cv2.imshow(window_name, frame)

        cv2.imshow(window_name, frame)
        cv2.setMouseCallback(window_name, mouse_cb)
        
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == 27 or key == ord('q'): break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
            if len(points) == 2:
                cv2.waitKey(1000) 
                break
                
        cv2.destroyAllWindows()
        
        if len(points) == 2:
            pt1 = np.array([[points[0]]], dtype=np.float32)
            pt2 = np.array([[points[1]]], dtype=np.float32)
            
            # Sử dụng GeometryUtils thay vì tự tính toán
            pt1_bev = cv2.perspectiveTransform(pt1, self.geo_utils.matrix)[0][0]
            pt2_bev = cv2.perspectiveTransform(pt2, self.geo_utils.matrix)[0][0]
            dist_px = self.geo_utils.calculate_pixel_distance(pt1_bev, pt2_bev)
            
            self.cfg_manager.pixels_per_meter = dist_px / ref_length
            self.cfg_manager.save_config() 
            self.log_alert(f"✅ Đã lưu tỷ lệ mới: 1m = {self.cfg_manager.pixels_per_meter:.1f}px")
        else:
            self.log_alert("❌ Đã hủy cân chỉnh tỷ lệ.")
            
        if was_running: 
            if self.source_type == 'video': self.load_video()
            elif self.source_type == 'camera': self.start_camera()
            elif self.source_type == 'image': self.load_image()

    # ==========================================
    # LUỒNG CHÍNH: XỬ LÝ HÌNH ẢNH & RADAR
    # ==========================================
    def update_video(self):
        if not self.is_running:
            return

        if self.ai.model is None:
            self.log_alert("❌ Chưa có Model nào được tải! Vui lòng chọn Model.")
            self.stop_camera()
            return

        if self.source_type == 'image':
            frame = cv2.imread(self.image_path)
            if frame is None:
                self.log_alert("❌ Lỗi đọc ảnh!")
                self.stop_camera()
                return
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.after(100, self.update_video)
                return

        # Lấy thông số từ ConfigManager
        danger_px = self.cfg_manager.danger_dist_m * self.cfg_manager.pixels_per_meter
        warning_px = self.cfg_manager.warning_dist_m * self.cfg_manager.pixels_per_meter

        # Gọi qua AIEngine
        results = self.ai.detect(frame, self.conf_threshold)
        annotated_frame = frame.copy()
        
        if self.radar_visible:
            bev_map = np.zeros((1000, 1000, 3), dtype=np.uint8) 
            for i in range(0, 1000, 100): cv2.line(bev_map, (i, 0), (i, 1000), (0, 50, 0), 1)
            for i in range(0, 1000, 100): cv2.line(bev_map, (0, i), (1000, i), (0, 50, 0), 1)
        
        forklifts = []
        persons = []
        current_alert_level = 0 

        if results is not None and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.ai.model.names[int(box.cls[0])]
                
                # Gọi qua GeometryUtils
                point = self.geo_utils.get_bottom_center(x1, y1, x2, y2)
                
                if class_name in ['Forklift', 'forklift']:
                    forklifts.append({'point': point})
                    cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.circle(annotated_frame, point, 5, (0,255,0), -1) 
                elif class_name in ['Person', 'person']:
                    persons.append({'point': point})
                    cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.circle(annotated_frame, point, 5, (0,0,255), -1) 

        for f in forklifts:
            f_bev = self.geo_utils.get_bev_point(f['point'])
            if self.radar_visible:
                cv2.circle(bev_map, f_bev, 20, (0, 255, 0), -1)
                cv2.circle(bev_map, f_bev, int(warning_px), (0, 255, 255), 2) 
                cv2.circle(bev_map, f_bev, int(danger_px), (0, 0, 255), 2)    
            
            for p in persons:
                p_bev = self.geo_utils.get_bev_point(p['point'])
                if self.radar_visible:
                    cv2.circle(bev_map, p_bev, 12, (0, 0, 255), -1)
                
                dist_px = self.geo_utils.calculate_pixel_distance(f_bev, p_bev)
                dist_m = dist_px / self.cfg_manager.pixels_per_meter
                
                if dist_px <= danger_px:
                    current_alert_level = max(current_alert_level, 2)
                    color = (0, 0, 255) 
                elif dist_px <= warning_px:
                    current_alert_level = max(current_alert_level, 1)
                    color = (0, 255, 255) 
                else:
                    color = (0, 255, 0) 
                
                cv2.line(annotated_frame, f['point'], p['point'], color, 2)
                mid_point = (int((f['point'][0] + p['point'][0])/2), int((f['point'][1] + p['point'][1])/2) - 10)
                cv2.putText(annotated_frame, f"{dist_m:.1f}m", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                
                if self.radar_visible:
                    cv2.line(bev_map, f_bev, p_bev, color, 3) 

        # XỬ LÝ GIAO DIỆN CẢNH BÁO
        current_time = time.time()
        if current_alert_level == 2:
            self.status_label.configure(text="NGUY HIỂM: SẮP VA CHẠM!", text_color="red")
            cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0,0,255), 10)
            if self.radar_visible:
                cv2.rectangle(bev_map, (0,0), (bev_map.shape[1], bev_map.shape[0]), (0,0,255), 20)
            if current_time - self.last_log_time > 5:
                self.log_alert(f"🚨 NGUY HIỂM! Vi phạm khoảng cách (< {self.cfg_manager.danger_dist_m}m).")
                self.last_log_time = current_time
        elif current_alert_level == 1:
            self.status_label.configure(text="CẢNH BÁO: CHÚ Ý QUAN SÁT", text_color="#ffcc00")
            cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0,255,255), 10)
            if self.radar_visible:
                cv2.rectangle(bev_map, (0,0), (bev_map.shape[1], bev_map.shape[0]), (0,255,255), 20)
            if current_time - self.last_log_time > 5:
                self.log_alert(f"⚠️ Cảnh báo: Có người tiếp cận (< {self.cfg_manager.warning_dist_m}m).")
                self.last_log_time = current_time
        else:
            if self.source_type != 'image':
                self.status_label.configure(text="Trạng thái: AN TOÀN", text_color="green")
  
        # XUẤT VIDEO 
        if self.camera_visible:
            v_width = self.video_label.winfo_width()
            v_height = self.video_label.winfo_height()
            rgb_video = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_video = Image.fromarray(rgb_video)
            
            if v_width > 10 and v_height > 10:
                f_h, f_w = annotated_frame.shape[:2]
                scale = min(v_width / f_w, v_height / f_h)
                new_w, new_h = int(f_w * scale), int(f_h * scale)
                img_video = img_video.resize((new_w, new_h))
            else:
                img_video = img_video.resize((960, 540))
                
            imgtk_video = ImageTk.PhotoImage(image=img_video)
            self.video_label.imgtk = imgtk_video
            self.video_label.configure(image=imgtk_video, text="")

        # XUẤT RADAR 
        if self.radar_visible:
            r_width = self.radar_label.winfo_width()
            r_height = self.radar_label.winfo_height()
            rgb_radar = cv2.cvtColor(bev_map, cv2.COLOR_BGR2RGB)
            img_radar = Image.fromarray(rgb_radar)
            
            if r_width > 10 and r_height > 10:
                b_h, b_w = bev_map.shape[:2] 
                scale = min(r_width / b_w, r_height / b_h)
                new_w, new_h = int(b_w * scale), int(b_h * scale)
                img_radar = img_radar.resize((new_w, new_h))
            else:
                img_radar = img_radar.resize((800, 800))
                
            imgtk_radar = ImageTk.PhotoImage(image=img_radar)
            self.radar_label.imgtk = imgtk_radar
            self.radar_label.configure(image=imgtk_radar)

        if self.source_type != 'image':
            self.after(10, self.update_video)