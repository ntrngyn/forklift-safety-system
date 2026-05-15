# 🚧 Forklift Safety Monitoring System

**Hệ thống giám sát an toàn và cảnh báo nguy cơ va chạm giữa người và xe nâng trong nhà xưởng.**

Ứng dụng công nghệ Thị giác máy tính (Computer Vision) và Học sâu (Deep Learning) để nhận diện, theo dõi và cảnh báo các tình huống nguy hiểm giữa người lao động và xe nâng trong thời gian thực tại môi trường nhà xưởng.

## ✨ Tính năng chính

- **Nhận diện thời gian thực:** Phát hiện chính xác người và xe nâng trong khung hình camera bằng mô hình Deep Learning (YOLO).
- **Đo lường & Ước tính khoảng cách:** Tính toán khoảng cách tương đối giữa người và xe nâng trong không gian xưởng.
- **Cảnh báo thông minh:** Tự động phát tín hiệu cảnh báo (trực quan trên màn hình hoặc âm thanh) khi khoảng cách vi phạm vùng an toàn đã thiết lập.
- **Giao diện thân thiện:** Cung cấp giao diện người dùng (UI) trực quan để giám sát camera và cấu hình hệ thống.

## 🛠 Công nghệ sử dụng

- **Ngôn ngữ:** Python
- **Computer Vision:** OpenCV, YOLOv8 (Object Detection)
- **Kiến trúc:** Phân tách rõ ràng giữa Core xử lý AI và User Interface.

## 📁 Cấu trúc dự án

Dự án được tổ chức theo module để dễ dàng mở rộng và bảo trì:

```text
forklift-safety-system/
├── core/          # Chứa logic xử lý chính (Load model, Inference, tính toán khoảng cách, logic cảnh báo)
├── ui/            # Mã nguồn giao diện người dùng (User Interface)
├── weights/       # Thư mục chứa các file trọng số (weights) của mô hình AI đã được huấn luyện (.pt)
├── config.json    # File cấu hình các thông số hệ thống (ngưỡng cảnh báo, vùng an toàn, thông số camera...)
└── main.py        # Điểm entry-point khởi chạy toàn bộ ứng dụng
```
