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
    noise_tolerance = max(2, int(box_width * 0.015)) 
    
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

def sort_reading_order(boxes, y_tolerance=15):
    """Sắp xếp theo thứ tự đọc: Từ trên xuống dưới, trái qua phải"""
    boxes.sort(key=lambda b: b['box'][1])
    
    lines = []
    current_line = []
    
    for box in boxes:
        if not current_line:
            current_line.append(box)
        else:
            y_baseline = current_line[0]['box'][1]
            y_current = box['box'][1]
            
            if abs(y_current - y_baseline) <= y_tolerance:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
                
    if current_line:
        lines.append(current_line)
        
    sorted_boxes = []
    for line in lines:
        line.sort(key=lambda b: b['box'][0])
        sorted_boxes.extend(line)
        
    return sorted_boxes

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

def smart_crop(img, box, pad_ratio=0.05, min_pad=2):
    x1, y1, x2, y2 = box
    h_img, w_img = img.shape[:2]
    
    # Cắt thô
    crop_raw = img[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
    
    # CHỐT CHẶN 1: Nếu ảnh cắt ra bị rỗng -> Bỏ qua ngay
    if crop_raw is None or crop_raw.size == 0:
        return None
    
    gray = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    
    if coords is None: 
        return crop_raw
    
    cx, cy, cw, ch = cv2.boundingRect(coords)
    pad = max(min_pad, int(ch * pad_ratio))
    
    f_x1 = max(0, x1 + cx - pad)
    f_y1 = max(0, y1 + cy - pad)
    f_x2 = min(w_img, x1 + cx + cw + pad)
    f_y2 = min(h_img, y1 + cy + ch + pad)
    
    # Cắt tinh
    final_crop = img[f_y1:f_y2, f_x1:f_x2]
    
    # CHỐT CHẶN 2: Đảm bảo kết quả cuối không rỗng
    if final_crop.size == 0:
        return None
        
    return final_crop

def enhance_image(img, target_height=32, pad_size=15):
    h, w = img.shape[:2]
    
    # CHỐT CHẶN 3: Không xử lý ảnh lỗi
    if h == 0 or w == 0:
        return img
        
    scale = target_height / float(h)
    
    # CHỐT CHẶN 4: Đảm bảo new_w ít nhất là 1 pixel (chống lỗi làm tròn xuống 0)
    new_w = max(1, int(w * scale))
    
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, target_height), interpolation=interp)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_img = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_img[norm_img > 200] = 255
    
    padded = cv2.copyMakeBorder(
        norm_img, 
        top=pad_size, bottom=pad_size, left=pad_size, right=pad_size, 
        borderType=cv2.BORDER_CONSTANT, 
        value=255
    )
    
    final_img = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    return final_img