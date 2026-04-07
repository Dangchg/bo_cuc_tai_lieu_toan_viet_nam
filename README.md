# 📚 Vietnamese Math Document OCR Pipeline

Dự án này là một Pipeline xử lý hình ảnh nâng cao, chuyên dụng để số hóa các tài liệu, đề thi Toán học tiếng Việt từ định dạng PDF hoặc ảnh chụp sang mã nguồn **LaTeX** chuẩn. Hệ thống kết hợp sức mạnh của **PaddleOCR** (Layout), **VietOCR** (Text) và **Pix2Tex** (Math) cùng với các thuật toán xử lý hình học tùy chỉnh.

## 🚀 Tính năng nổi bật

* **Đa dạng đầu vào:** Hỗ trợ cả ảnh đơn lẻ (JPG, PNG) và file PDF nhiều trang (tự động trích xuất bằng PyMuPDF).
* **Phân tích Layout thông minh:** Tự động nhận diện tiêu đề, đoạn văn, công thức toán học và hình ảnh minh họa.
* **Xử lý Inline Math:** Thuật toán tách dòng (Line Segmentation) và xử lý đè lấn (Overlap Resolution) giúp trích xuất chính xác công thức nằm giữa dòng chữ.
* **Smart Cropping:** Kỹ thuật "xén sát tận xương" (Auto-tighten) giúp loại bỏ khoảng trắng thừa, tối ưu hóa đầu vào cho các mô hình học sâu.
* **Image Enhancement:** Tự động tăng độ phân giải (Bicubic Upscaling), khử nhiễu và làm nét nét mực để tăng độ chính xác của OCR.
* **Đóng gói tự động:** Xuất kết quả dưới dạng file `.tex` và đóng gói toàn bộ tài nguyên (ảnh minh họa) vào file `.zip`.

## 🛠 Yêu cầu hệ thống

Cài đặt các thư viện cần thiết trước khi chạy:

```bash
pip install paddleocr vietocr pix2tex PyMuPDF opencv-python pillow
```

## 📂 Cấu trúc dự án

| File | Chức năng chính |
| :--- | :--- |
| `main.py` | Điều phối Pipeline, quản lý luồng dữ liệu và xuất kết quả (LaTeX, ZIP). |
| `models.py` | Khởi tạo và quản lý các mô hình AI (PaddleOCR, VietOCR, Pix2Tex). |
| `image_utils.py` | "Trái tim" hình học: Tách dòng, xử lý đè lấn, Smart Crop, tăng độ phân giải. |
| `debug_results/` | (Tự động tạo) Chứa ảnh bóc tách layout và các mảnh crop để kiểm tra lỗi. |

## ⚙️ Luồng xử lý kỹ thuật (Workflow)

1.  **PDF/Image Pre-processing:** Chuyển đổi PDF về 300 DPI để đảm bảo độ nét.
2.  **Layout Analysis:** PaddleOCR bóc tách các vùng (regions).
3.  **Coordinate Unscaling:** Đồng bộ tọa độ từ hệ quy chiếu của mô hình về ảnh gốc.
4.  **Geometry Processing:**
    * Tách các Paragraph thành từng dòng chữ riêng biệt.
    * Cắt dọc các dòng có chứa công thức toán để tách riêng mảnh Text và mảnh Math.
5.  **Smart Crop & Enhance:** Xén khít nét mực, phóng to 2x và chuẩn hóa tương phản.
6.  **AI Inference:** Chạy VietOCR cho chữ và Pix2Tex cho toán học.
7.  **Final Export:** Lắp ráp mã LaTeX và nén ZIP.

## 🔍 Chế độ Debug

Trong file `main.py`, đặt `DEBUG_MODE = True` để hệ thống xuất các file ảnh kiểm tra trong thư mục `./debug_results`:
* `1_raw_layout_pX.jpg`: Layout thô từ model.
* `2_final_order_pX.jpg`: Thứ tự đọc và các box đã được xử lý hình học.
* `/crops/`: Từng mảnh ảnh nhỏ được đưa vào mô hình OCR (dùng để soi lỗi lẹm chữ).

## 📝 Lưu ý
* Hệ thống yêu cầu kết nối mạng trong lần chạy đầu tiên để tải trọng số (weights) của các mô hình AI.
* Để đạt tốc độ tốt nhất, nên cấu hình sử dụng GPU (CUDA) trong file `models.py`.

---

