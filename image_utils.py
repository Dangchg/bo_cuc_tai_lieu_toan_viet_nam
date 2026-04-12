import fitz  # PyMuPDF
import numpy as np
import cv2
import os


def read_pdf_to_cv2_images(pdf_path, dpi=300):
    """
    Chuyển đổi file PDF thành danh sách các ảnh OpenCV (định dạng BGR).
    dpi=300 là chuẩn phân giải tối ưu cho bài toán OCR.
    """
    print(f"-> Đang trích xuất ảnh từ PDF (DPI: {dpi})...")
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Thiết lập tỷ lệ zoom để đạt được DPI mong muốn (mặc định PDF là 72 DPI)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        # Render trang PDF thành pixel (pixmap)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Chuyển đổi dữ liệu byte của pixmap thành ma trận numpy
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # PyMuPDF trả về hệ màu RGB, trong khi OpenCV dùng BGR
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        images.append(img_cv2)
        
    print(f"-> Đã trích xuất {len(images)} trang từ PDF.")
    return images

def split_paragraph_to_lines(img, text_box):
    x1, y1, x2, y2 = text_box
    
    # Ràng buộc tọa độ
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    
    crop_img = img[y1:y2, x1:x2]
    
    # CHỐT CHẶN 5: Nếu box bị ép thành 0 chiều
    if crop_img.size == 0:
        return []
        
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15, 5
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    horizontal_projection = np.sum(thresh, axis=1) / 255.0 
    
    line_boxes = []
    in_line = False
    start_y = 0
    min_line_height = 8
    
    box_width = x2 - x1
    noise_tolerance = max(2, int(box_width * 0.035)) 
    
    for y, val in enumerate(horizontal_projection):
        if val > noise_tolerance and not in_line:
            start_y = y
            in_line = True
        elif val <= noise_tolerance and in_line:
            end_y = y
            if end_y - start_y > min_line_height:
                line_boxes.append({
                    'label': 'text_line',
                    'box': [x1, y1 + start_y, x2, y1 + end_y]
                })
            in_line = False
            
    if in_line and (len(horizontal_projection) - start_y > min_line_height):
        line_boxes.append({
            'label': 'text_line',
            'box': [x1, y1 + start_y, x2, y2]
        })
        
    if not line_boxes:
        return [{'label': 'text_line', 'box': text_box}]
        
    return line_boxes

def resolve_overlaps(boxes, overlap_threshold=5):
    """Cắt box text nếu bị box formula đè lên"""
    resolved_boxes = []
    formulas = [b for b in boxes if b['label'] in ['formula', 'texformula']]
    texts = [b for b in boxes if b['label'] in ['text', 'text_line', 'paragraph_title', 'doc_title']]
    others = [b for b in boxes if b['label'] not in ['text', 'text_line', 'paragraph_title', 'doc_title', 'formula', 'texformula']]

    for t_box in texts:
        tx1, ty1, tx2, ty2 = t_box['box']
        overlapping_formulas = []

        for f_box in formulas:
            fx1, fy1, fx2, fy2 = f_box['box']
            if not (tx2 < fx1 or tx1 > fx2 or ty2 < fy1 or ty1 > fy2):
                overlapping_formulas.append(f_box)

        if not overlapping_formulas:
            resolved_boxes.append(t_box)
        else:
            overlapping_formulas.sort(key=lambda b: b['box'][0])
            current_tx1 = tx1
            
            for f_box in overlapping_formulas:
                fx1, fy1, fx2, fy2 = f_box['box']
                if fx1 > current_tx1 + overlap_threshold:
                    resolved_boxes.append({
                        'label': t_box['label'],
                        'box': [current_tx1, ty1, fx1, ty2]
                    })
                current_tx1 = fx2
                
            if current_tx1 < tx2 - overlap_threshold:
                resolved_boxes.append({
                    'label': t_box['label'],
                    'box': [current_tx1, ty1, tx2, ty2]
                })

    resolved_boxes.extend(formulas)
    resolved_boxes.extend(others)
    return resolved_boxes

def sort_reading_order(boxes):
    """
    Sắp xếp thứ tự đọc chuẩn: Từ trên xuống dưới, Trái qua phải.
    Sử dụng 'Dung sai trục Y' để gom các phần tử vào đúng dòng.
    """
    if not boxes: return []

    # Sắp xếp sơ bộ từ trên xuống dưới theo đỉnh Y
    boxes.sort(key=lambda x: x['box'][1])

    lines = []
    current_line = [boxes[0]]

    for box in boxes[1:]:
        base_box = current_line[0]['box']
        curr_box = box['box']

        # Tính dung sai: Mức chênh lệch cho phép bằng 50% chiều cao của box gốc
        tolerance = (base_box[3] - base_box[1]) * 0.5

        # Nếu Y của box hiện tại nằm trong dung sai -> Nó thuộc cùng 1 dòng
        if abs(curr_box[1] - base_box[1]) < tolerance:
            current_line.append(box)
        else:
            # Sang dòng mới: Sắp xếp dòng cũ từ trái qua phải (theo tọa độ X)
            current_line.sort(key=lambda x: x['box'][0])
            lines.extend(current_line)
            current_line = [box] # Khởi tạo dòng mới

    # Xử lý nốt dòng cuối cùng
    if current_line:
        current_line.sort(key=lambda x: x['box'][0])
        lines.extend(current_line)

    return lines

def process_layout_boxes(img, boxes):
    """Wrapper xử lý chuỗi: Cắt đoạn văn -> Xử lý đè lấn -> Sắp xếp"""
    refined_boxes = []
    formulas = [b for b in boxes if b['label'] in ['formula', 'texformula']]
    
    for box_info in boxes:
        if box_info['label'] in ['text', 'paragraph']:
            # Kiểm tra xem box này có đè lên formula nào không
            tx1, ty1, tx2, ty2 = box_info['box']
            has_inline_math = any(not (tx2 < f['box'][0] or tx1 > f['box'][2] or ty2 < f['box'][1] or ty1 > f['box'][3]) for f in formulas)
            
            if has_inline_math:
                # Nếu có đè -> Tách thành từng dòng nhỏ
                lines = split_paragraph_to_lines(img, box_info['box'])
                refined_boxes.extend(lines)
            else:
                # Không đè -> Giữ nguyên
                refined_boxes.append(box_info)
        else:
            refined_boxes.append(box_info)
            
    # Gọi hàm cắt lấn 
    final_boxes = resolve_overlaps(refined_boxes) 
    
    # Sắp xếp thứ tự đọc
    ordered_boxes = sort_reading_order(final_boxes)
    
    return ordered_boxes
def draw_boxes_on_image(image, boxes, output_path, draw_order=False):
    """Vẽ bounding boxes và label lên ảnh để kiểm tra"""
    debug_img = image.copy()
    colors = {
        'text': (0, 255, 255), 'text_line': (255, 0, 0), 
        'formula': (0, 255, 0), 'texformula': (0, 255, 0)
    }

    for i, item in enumerate(boxes):
        x1, y1, x2, y2 = item['box']
        label = item['label']
        color = colors.get(label, (200, 200, 200))
        
        # Vẽ box
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ label và thứ tự đọc
        text = f"[{i+1}] {label}" if draw_order else label
        cv2.putText(debug_img, text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(output_path, debug_img)

def unscale_layout_boxes(original_img, layout_regions, model_input_size=(1024, 1024)):
    """Sửa lỗi lệch tọa độ bằng cách map ngược về kích thước gốc"""
    h_goc, w_goc = original_img.shape[:2]
    w_model, h_model = model_input_size
    ratio_w, ratio_h = w_goc / w_model, h_goc / h_model
    
    for item in layout_regions:
        x1, y1, x2, y2 = item['box']
        item['box'] = [int(x1 * ratio_w), int(y1 * ratio_h), 
                        int(x2 * ratio_w), int(y2 * ratio_h)]
    return layout_regions

def smart_crop(img, box, pad_ratio=0.1, min_pad=5):
    """
    Phiên bản bảo vệ dấu tiếng Việt:
    - min_pad=5: Đảm bảo khoảng cách an toàn cho các dấu nằm sát mép.
    - Dùng ngưỡng Threshold thấp hơn để không bỏ sót các dấu mờ.
    """
    x1, y1, x2, y2 = box
    h_img, w_img = img.shape[:2]
    crop_raw = img[y1:y2, x1:x2]
    
    if crop_raw is None or crop_raw.size == 0: return None
    
    gray = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2GRAY)
    
    # Thay đổi: Giảm ngưỡng từ 200 xuống khoảng 180 hoặc dùng Otsu 
    # để bắt được cả các dấu phụ có nét mực nhạt
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    coords = cv2.findNonZero(thresh)
    if coords is None: return crop_raw
    
    cx, cy, cw, ch = cv2.boundingRect(coords)
    
    # Tăng pad_ratio lên 0.1 (10%) và min_pad lên 5 để tránh việc 
    # nét vẽ khung Debug hoặc sự sai số làm tròn "liếm" mất dấu.
    pad = max(min_pad, int(ch * pad_ratio))
    
    f_x1 = max(0, x1 + cx - pad)
    f_y1 = max(0, y1 + cy - pad)
    f_x2 = min(w_img, x1 + cx + cw + pad)
    f_y2 = min(h_img, y1 + cy + ch + pad)
    
    return img[f_y1:f_y2, f_x1:f_x2]
def enhance_text_for_vietocr(img, target_height=32, pad_size=15):
    """
    Tối ưu hóa ảnh chữ tiếng Việt cho VietOCR:
    1. Ép về chiều cao chuẩn 32px.
    2. Chuẩn hóa dải màu (giữ độ mượt của viền chữ).
    3. Thêm đệm trắng an toàn.
    """
    h, w = img.shape[:2]
    
    # Chốt chặn an toàn cho ma trận rỗng
    if h == 0 or w == 0:
        return img
        
    # 1. Resize về đúng chiều cao huấn luyện
    scale = target_height / float(h)
    new_w = max(1, int(w * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, target_height), interpolation=interp)
    
    # 2. Chuyển xám và Kéo giãn tương phản (Normalization)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_img = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Làm sạch nền: Pixel xám nhạt (bóng mờ) sẽ bị ép thành trắng tinh
    norm_img[norm_img > 200] = 255
    
    # 3. Bọc thêm viền trắng tinh ở cả 4 cạnh
    padded = cv2.copyMakeBorder(
        norm_img, 
        top=pad_size, bottom=pad_size, left=pad_size, right=pad_size, 
        borderType=cv2.BORDER_CONSTANT, 
        value=255 
    )
    
    # Chuyển lại hệ BGR để Model đọc
    final_img = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    return final_img

def enhance_math_for_pix2tex(img, scale_factor=2.0, pad_size=15):
    """
    Tối ưu hóa ảnh công thức toán cho Pix2Tex:
    1. Upscale 2x để làm rõ các chỉ số (subscript/superscript).
    2. Áp dụng ma trận làm nét (Sharpen).
    3. Nhị phân hóa sâu (Otsu) để loại bỏ nhiễu lưới.
    """
    h, w = img.shape[:2]
    
    # Chốt chặn an toàn
    if h == 0 or w == 0:
        return img
        
    # 1. Phóng to ảnh (Sử dụng CUBIC để giữ viền nét mực cong)
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 2. Khử nhiễu nhẹ trước khi làm nét
    blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
    
    # 3. Áp dụng Kernel làm nét mạnh
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
    
    # 4. Nhị phân hóa: Ép về 2 màu Trắng/Đen tuyệt đối để Pix2Tex bắt nét
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Thêm viền trắng (Pix2Tex rất dễ đoán sai nếu ngoặc vuông/tròn bị sát mép ảnh)
    padded = cv2.copyMakeBorder(
        thresh, 
        top=pad_size, bottom=pad_size, left=pad_size, right=pad_size, 
        borderType=cv2.BORDER_CONSTANT, 
        value=255
    )
    
    final_img = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    return final_img

def resolve_inline_math(raw_regions):
    """
    Tìm công thức nằm trong dòng, chẻ nhỏ box text và chống lặp công thức.
    ĐÃ TỐI ƯU: Thêm chốt chặn trục X và chống lẹm khung.
    """
    final_regions = []
    
    # Gộp chung các nhãn text
    text_labels = ['text_line', 'doc_title', 'paragraph_title']
    text_lines = [b for b in raw_regions if b['label'] in text_labels]
    formulas = [b for b in raw_regions if b['label'] in ['formula', 'texformula']]
    
    # Giữ nguyên các box không phải text hay math
    other_regions = [b for b in raw_regions if b['label'] not in text_labels + ['formula', 'texformula']]

    used_formulas = set()

    for line in text_lines:
        lx1, ly1, lx2, ly2 = line['box']
        line_height = ly2 - ly1
        
        inline_math_in_this_line = []
        for i, f in enumerate(formulas):
            if i in used_formulas: 
                continue
                
            fx1, fy1, fx2, fy2 = f['box']
            f_height = fy2 - fy1
            overlap_y = min(ly2, fy2) - max(ly1, fy1)
            
            # --- TỐI ƯU Ở ĐÂY ---
            # 1. Đủ độ giao cắt Y (> 40% chiều cao chữ)
            # 2. Không quá cao (< 2.5 lần chiều cao chữ)
            # 3. CHỐT CHẶN TRỤC X: Tọa độ công thức phải nằm xen vào giữa dòng chữ
            if overlap_y > (line_height * 0.4) and f_height < (line_height * 2.5):
                if fx1 < lx2 and fx2 > lx1: 
                    inline_math_in_this_line.append((i, f))
                    used_formulas.add(i)

        if not inline_math_in_this_line:
            final_regions.append(line)
        else:
            inline_math_in_this_line.sort(key=lambda item: item[1]['box'][0])
            current_x = lx1
            MIN_TEXT_CHUNK_WIDTH = 20 

            for idx, math_box in inline_math_in_this_line:
                fx1, fy1, fx2, fy2 = math_box['box']
                
                # --- TỐI ƯU Ở ĐÂY: Dùng min(fx1, lx2) để chữ bên trái không đâm xuyên qua lề phải ---
                chunk_right = min(fx1, lx2)
                
                if current_x < chunk_right and (chunk_right - current_x) > MIN_TEXT_CHUNK_WIDTH:
                    final_regions.append({
                        'label': line.get('label', 'text_line'),
                        'box': [current_x, ly1, chunk_right, ly2]
                    })
                
                final_regions.append(math_box)
                current_x = max(current_x, fx2 + 2)
                
            if current_x < lx2 and (lx2 - current_x) > MIN_TEXT_CHUNK_WIDTH:
                final_regions.append({
                    'label': line.get('label', 'text_line'),
                    'box': [current_x, ly1, lx2, ly2]
                })

    for i, f in enumerate(formulas):
        if i not in used_formulas:
            final_regions.append(f)
            
    final_regions.extend(other_regions)
    
    # LƯU Ý: Đảm bảo bạn đã import/định nghĩa hàm sort_reading_order
    return sort_reading_order(final_regions)

def calculate_intersection_area(box1, box2):
    """Tính diện tích phần giao nhau giữa 2 bounding boxes [x1, y1, x2, y2]"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)

def remove_overlapping_text_lines(regions, overlap_threshold=0.5):
    """
    Lọc bỏ mọi dòng chữ (text_line, doc_title, paragraph_title) đè lên ảnh/công thức.
    ĐÃ TỐI ƯU: Xử lý triệt để tất cả các nhãn văn bản.
    """
    priority_labels = ['image', 'figure', 'formula', 'texformula']
    priority_boxes = [r['box'] for r in regions if r['label'] in priority_labels]
    
    # --- TỐI ƯU Ở ĐÂY: Quét mọi nhãn chữ ---
    text_labels = ['text_line', 'doc_title', 'paragraph_title']
    
    filtered_regions = []
    
    for item in regions:
        # Nếu là nhãn ưu tiên hoặc nhãn bảng biểu, auto giữ lại
        if item['label'] not in text_labels:
            filtered_regions.append(item)
            continue
            
        # Nếu là nhãn text, tiến hành xét duyệt IoA
        t_box = item['box']
        t_area = (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])
        
        if t_area <= 0: 
            continue

        should_keep = True
        for p_box in priority_boxes:
            inter_area = calculate_intersection_area(t_box, p_box)
            
            # Nếu phần chữ bị chìm trong ảnh/công thức vượt ngưỡng -> Khai tử
            if (inter_area / t_area) > overlap_threshold:
                should_keep = False
                break 

        if should_keep:
            filtered_regions.append(item)

    return filtered_regions

# Hàm helper của bạn rất chuẩn, mình giữ nguyên:
def calculate_intersection_area(box1, box2):
    """Tính diện tích phần giao nhau giữa 2 bounding boxes [x1, y1, x2, y2]"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)

def remove_watermark(img):
    """
    Xóa watermark bằng kỹ thuật Cắt sáng (Intensity Truncation).
    Chuyên trị các watermark chữ to, bóng mờ chìm dưới nền giấy.
    """
    if img is None or img.size == 0:
        return np.ones((5, 5, 3), dtype=np.uint8) * 255

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. CHÉM BẰNG (TRUNCATE)
        # Bất kỳ pixel nào sáng hơn 180 (watermark nhạt và nền trắng) đều bị ép về 180.
        # Nét chữ thật (màu đen, pixel < 100) được giữ nguyên.
        # MẸO: Nếu watermark đậm hơn một chút, bạn có thể hạ số 180 xuống 160 hoặc 150.
        _, trunc = cv2.threshold(gray, 180, 255, cv2.THRESH_TRUNC)

        # 2. KÉO DÃN (NORMALIZE)
        # Kéo dải màu hiện tại: ép màu xám (180) thành trắng tinh (255), nét đen vẫn đen.
        normalized = cv2.normalize(trunc, None, 0, 255, cv2.NORM_MINMAX)

        # 3. Phân ngưỡng Otsu để lấy nét mực dứt khoát
        _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 4. Áp mask xóa nền
        result = img.copy()
        result[binary == 255] = [255, 255, 255]

        return result

    except Exception as e:
        print(f"[Cảnh báo] Lỗi thuật toán xóa watermark: {e}")
        return img
def is_valid_image_content(img_crop):
    """
    Kiểm tra xem ảnh crop có chứa nội dung thật không, hay chỉ là rác/bóng mờ/đường kẻ.
    """
    if img_crop is None or img_crop.size == 0:
        return False

    h, w = img_crop.shape[:2]

    # 1. KÍCH THƯỚC TỐI THIỂU
    if h < 15 or w < 15:
        return False

    # 2. LỌC ĐƯỜNG KẺ BẰNG TỶ LỆ KHUNG HÌNH (Aspect Ratio)
    # Rất nhiều rác là các đường kẻ dọc/ngang (w gấp 60 lần h, hoặc h gấp 20 lần w)
    aspect_ratio = w / float(h)
    if aspect_ratio > 60 or aspect_ratio < 0.05:
        return False

    # 3. LỌC BÓNG MỜ BẰNG ĐỘ SẮC NÉT (Laplacian Variance)
    # Chữ/Toán luôn có cạnh sắc nét. Rác mờ sẽ có phương sai rất thấp.
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Ngưỡng 30-50 là mức cơ bản để lọc các mảng mờ tịt không có chi tiết sắc cạnh
    if variance < 30: 
        return False

    return True

def nms_text_lines_by_score(regions, overlap_threshold=0.5):
    """
    Thuật toán Non-Maximum Suppression (NMS) tùy chỉnh cho OCR.
    Phát hiện các text_line lồng nhau và GIỮ LẠI box có SCORE (điểm) CAO HƠN.
    """
    # 1. Tách nhóm text ra để lọc (không đụng chạm đến table, image, formula)
    text_labels = ['text_line', 'doc_title', 'paragraph_title']
    text_regions = [r for r in regions if r['label'] in text_labels]
    other_regions = [r for r in regions if r['label'] not in text_labels]

    # 2. Sắp xếp các text_line theo điểm `score` giảm dần (Từ cao xuống thấp)
    # Box nào có điểm cao nhất sẽ luôn được duyệt đầu tiên
    text_regions.sort(key=lambda x: x.get('score', 1.0), reverse=True)

    keep_regions = []
    
    while len(text_regions) > 0:
        # Lấy box có điểm CAO NHẤT hiện tại ra khỏi mảng và giữ lại nó
        current = text_regions.pop(0)
        keep_regions.append(current)

        remaining = []
        for other in text_regions:
            box1 = current['box']
            box2 = other['box']

            # Tính tọa độ phần giao nhau
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])

            # Nếu có giao nhau
            if x_right > x_left and y_bottom > y_top:
                inter_area = (x_right - x_left) * (y_bottom - y_top)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                # Dùng IoM: So sánh phần giao nhau với diện tích của box NHỎ HƠN
                min_area = min(area1, area2)
                overlap_ratio = inter_area / min_area if min_area > 0 else 0

                # Nếu lồng nhau quá ngưỡng -> Box kia bị coi là thừa (do điểm thấp hơn) -> XÓA
                if overlap_ratio > overlap_threshold:
                    continue # Không append vào mảng remaining nữa

            # Nếu không lồng nhau hoặc lồng rất ít -> Giữ lại để xét tiếp
            remaining.append(other)
            
        # Cập nhật lại danh sách chờ
        text_regions = remaining

    # 3. Trả về danh sách text đã lọc sạch sẽ + các vùng khác (bảng, ảnh, toán)
    return keep_regions + other_regions
