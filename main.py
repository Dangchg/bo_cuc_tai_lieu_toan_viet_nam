import cv2
import os
import shutil
from models import get_layout_regions, predict_vietocr, predict_math_latex
from image_utils import (
    smart_crop, 
    draw_boxes_on_image, 
    read_pdf_to_cv2_images, 
    resolve_inline_math,        
    enhance_text_for_vietocr,   
    enhance_math_for_pix2tex,   
    remove_overlapping_text_lines,remove_watermark,is_valid_image_content,nms_text_lines_by_score
)

from table_handler import assign_lines_to_cells, build_latex_table

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
DEBUG_MODE = True
DEBUG_DIR = "./debug_results"
EXPORT_DIR = "./ket_qua_xuat"
ZIP_FILENAME = "Tai_Lieu_So_Hoa"

def export_to_latex(content_list, output_filename="output.tex"):
    """Đóng gói dữ liệu thành file LaTeX chuẩn có kiểm tra cùng dòng."""
    latex_header = r"""\documentclass[10pt]{article}
\usepackage[vietnamese]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T5]{fontenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage[version=4]{mhchem}
\usepackage{stmaryrd}
\usepackage{graphicx}
\usepackage[export]{adjustbox}

\DeclareUnicodeCharacter{25A1}{\ifmmode\square\else{$\square$}\fi}

\begin{document}
"""
    latex_footer = r"\end{document}"
    
    body_content = ""
    prev_box = None
    prev_label = None 
    
    for item in content_list:
        if isinstance(item, str):
            body_content += "\n\n" + item + "\n\n"
            prev_box = None 
            prev_label = None
            continue
            
        text = item['text']
        curr_box = item['box']
        label = item.get('label', 'text_line') # Lấy nhãn ra
        
        # 1. BỌC CẤU TRÚC LATEX DỰA TRÊN NHÃN
        if label == 'doc_title':
            text = f"\\section*{{{text}}}"
        elif label == 'paragraph_title':
            text = f"\\subsection*{{{text}}}"
        elif label == 'header': 
            text = f"\\section*{{{text}}}"
        
        # 2. LOGIC NỐI DÒNG / XUỐNG DÒNG
        if prev_box is None:
            body_content += text
        else:
            # ÉP XUỐNG DÒNG: Nếu phần tử hiện tại HOẶC phần tử trước đó là đối tượng Độc Lập 
            # (Tiêu đề, Bảng, Ảnh minh họa) -> Tuyệt đối không nối dòng, phải tách riêng.
            independent_labels = ['doc_title', 'paragraph_title', 'table', 'image', 'figure']
            
            if label in independent_labels or prev_label in independent_labels:
                body_content += "\n\n" + text
            else:
                # KIỂM TRA DUNG SAI (Chỉ áp dụng cho text_line và formula)
                prev_h = prev_box[3] - prev_box[1]
                tolerance = prev_h * 0.5
                
                if abs(curr_box[1] - prev_box[1]) < tolerance:
                    # Nằm cùng dòng -> Nối bằng khoảng trắng
                    body_content += " " + text
                else:
                    # Khác dòng -> Xuống đoạn
                    body_content += "\n\n" + text
                
        prev_box = curr_box
        prev_label = label

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(latex_header)
        f.write(body_content)
        f.write("\n" + latex_footer)
        
    print(f"-> Đã xuất file thành công: {output_filename}")


def process_single_image(img_goc, page_index=1):
    """Lõi xử lý hình học và OCR cho một trang tài liệu."""
    results = []
    img_remove = remove_watermark(img_goc)
    # 1. Lưu ảnh tạm để PPStructureV3 đọc
    temp_img_path = f"temp_page_{page_index}.jpg"
    cv2.imwrite(temp_img_path, img_remove)
    
    print(f"--- Đang phân tích Layout trang {page_index} ---")
    # raw_regions lúc này ĐÃ BAO GỒM cả text_line và formula
    raw_regions = get_layout_regions(temp_img_path)
    #  HÀM LỌC 
    MIN_LAYOUT_SCORE = 0.3
    raw_regions = [b for b in raw_regions if b.get('score', 1.0) >= MIN_LAYOUT_SCORE]
    print(f"-> Đã lọc bỏ các vùng có độ tin cậy < {MIN_LAYOUT_SCORE}")

    print("-> Áp dụng NMS lọc các dòng chữ lồng nhau...")
    raw_regions = nms_text_lines_by_score(raw_regions, overlap_threshold=0.5)

    print("-> Lọc bỏ các box chữ rác nằm trong ảnh/công thức...")
    cleaned_regions = remove_overlapping_text_lines(raw_regions, overlap_threshold=0.4) 
    # Mình để 0.4 (40%) vì đôi khi box image không khoanh hết chữ, để ngưỡng thấp sẽ dọn rác sạch hơn.



    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    # [DEBUG 1] Ảnh thô từ model
    if DEBUG_MODE:
        draw_boxes_on_image(img_goc, raw_regions, os.path.join(DEBUG_DIR, f"1_raw_layout_p{page_index}.jpg"))

    # 2. Xử lý hình học: Ghép công thức vào giữa dòng chữ
    print("-> Xử lý Inline Math và Thứ tự đọc...")
    ordered_boxes = resolve_inline_math(cleaned_regions)
    
    # [DEBUG 2] Ảnh sau khi đã sắp xếp và chẻ dòng
    if DEBUG_MODE:
        draw_boxes_on_image(img_goc, ordered_boxes, os.path.join(DEBUG_DIR, f"2_final_order_p{page_index}.jpg"), draw_order=True)

    # 3. Phân luồng Cắt ảnh và nhận diện AI
    i = 0
    while i < len(ordered_boxes):
        item = ordered_boxes[i]
        label = item['label']
        x1, y1, x2, y2 = item['box']
        
        # --- LUỒNG 0: XỬ LÝ BẢNG ---
        if label == 'table':
            table_box = item['box']
            cell_boxes = item.get('cell_box_list', [])
            
            if not cell_boxes:
                i += 1
                continue
                
            cells_data, remaining_boxes = assign_lines_to_cells(table_box, cell_boxes, ordered_boxes)
            
            # Reset mảng duyệt bằng các box còn lại (đã loại bỏ chữ trong bảng)
            ordered_boxes = remaining_boxes
            
            cell_text_results = []
            for cell_idx in range(len(cell_boxes)):
                cell_content_list = cells_data[cell_idx]['contents']
                cell_combined_text = []
                for c_item in cell_content_list:
                    c_crop = smart_crop(img_goc, c_item['box'], pad_ratio=0.1)
                    
                    # SỬ DỤNG HÀM KIỂM ĐỊNH MỚI
                    if not is_valid_image_content(c_crop): 
                        continue
                        
                    if c_item['label'] in ['formula', 'texformula']:
                        res = predict_math_latex(enhance_math_for_pix2tex(c_crop))
                    else:
                        res = predict_vietocr(enhance_text_for_vietocr(c_crop))
                        
                    # LỚP BẢO VỆ CUỐI CÙNG: Nếu OCR trả ra rỗng hoặc 1 dấu chấm/phẩy thì bỏ qua
                    if len(res.strip()) <= 1 and res.strip() in ["", ".", ",", "-", "_", "|", "~", ":", "'", '"']:
                        continue
                        
                    cell_combined_text.append(res)

                    
                    
                cell_text_results.append(" ".join(cell_combined_text))
                
            latex_table = build_latex_table(cell_text_results, cell_boxes)
            results.append({'text': latex_table, 'box': table_box,'label': label})
            

            continue

        # --- LUỒNG 1: ẢNH MINH HỌA (IMAGE/FIGURE) ---
        elif label in ['image', 'figure']:
            pad_y = max(15, int((y2 - y1) * 0.05))
            pad_x = max(15, int((x2 - x1) * 0.05))
            c_y1, c_y2 = max(0, y1 - pad_y), min(img_goc.shape[0], y2 + pad_y)
            c_x1, c_x2 = max(0, x1 - pad_x), min(img_goc.shape[1], x2 + pad_x)
            
            figure_crop = img_goc[c_y1:c_y2, c_x1:c_x2]
            if figure_crop.size > 0:
                img_name = f"hinh_minh_hoa_p{page_index}_{i+1}.jpg"
                cv2.imwrite(os.path.join(EXPORT_DIR, img_name), figure_crop)
                
                latex_img_code = f"\\includegraphics[max width=0.8\\textwidth, center]{{{img_name}}}"
                results.append({'text': latex_img_code, 'box': [x1, y1, x2, y2], 'label': label})

        # --- LUỒNG 2: VĂN BẢN (TEXT LINE) ---
        elif label in ['text_line', 'doc_title', 'paragraph_title']:
            h_box = y2 - y1
            expand_v = int(h_box * 0)
            expanded_box = [x1, max(0, y1 - expand_v), x2, min(img_goc.shape[0], y2 + expand_v)]
            
            crop = smart_crop(img_goc, expanded_box, pad_ratio=0, min_pad=1)
            # SỬ DỤNG HÀM KIỂM ĐỊNH MỚI
            if is_valid_image_content(crop):
                if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_DIR, "crops", f"part_p{page_index}_{i+1:03d}_text.jpg"), crop)
                enhanced_crop = enhance_text_for_vietocr(crop)
                text_res = predict_vietocr(enhanced_crop)
                
                # Lọc rác chữ
                if len(text_res.strip()) > 1 or text_res.strip() not in ["", ".", ",", "-", "_", "|", "~", ":", "'", '"']:
                    results.append({'text': text_res, 'box': [x1, y1, x2, y2], 'label': label})

        # --- LUỒNG 3: TOÁN HỌC (FORMULA) ---
        elif label in ['formula', 'texformula']:
            crop = smart_crop(img_goc, [x1, y1, x2, y2], pad_ratio=0.05, min_pad=2)
            if crop is not None and crop.size > 0:
                if is_valid_image_content(crop):
                    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_DIR, "crops", f"part_p{page_index}_{i+1:03d}_math.jpg"), crop)
                    enhanced_crop = enhance_math_for_pix2tex(crop)
                    math_res = predict_math_latex(enhanced_crop)
                    
                    # Bỏ qua nếu OCR trả rỗng hoặc chỉ có cái vỏ bọc $...$
                    if math_res.strip() not in ["", "$$", "$ $"]:
                        results.append({'text': math_res, 'box': [x1, y1, x2, y2], 'label': label})

        i += 1

    return results


def main_pipeline(input_path):
    # Khởi tạo thư mục
    if DEBUG_MODE:
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(os.path.join(DEBUG_DIR, "crops"))
        
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR)
    
    # Phân loại đầu vào (PDF hoặc Ảnh)
    images_to_process = []
    if input_path.lower().endswith('.pdf'):
        images_to_process = read_pdf_to_cv2_images(input_path, dpi=300)
    else:
        img_cv = cv2.imread(input_path)
        if img_cv is None: raise ValueError(f"Lỗi: Không thể đọc ảnh tại {input_path}")
        images_to_process.append(img_cv)
        
    all_results = []
    
    # Xử lý từng trang
    for page_idx, img in enumerate(images_to_process):
        page_results = process_single_image(img, page_index=page_idx+1)
        all_results.extend(page_results)
        # Thêm ngắt trang sau mỗi page
        all_results.append(r"\newpage")
        
    # Xuất LaTeX
    print("\n-> Đang khởi tạo file LaTeX tổng...")
    if all_results and all_results[-1] == r"\newpage":
        all_results.pop() # Bỏ dấu ngắt trang thừa ở trang cuối
        
    tex_filepath = os.path.join(EXPORT_DIR, "ket_qua_bai_toan.tex")
    export_to_latex(all_results, tex_filepath)
    
    # Nén ZIP
    print(f"-> Đang đóng gói dữ liệu thành {ZIP_FILENAME}.zip ...")
    shutil.make_archive(ZIP_FILENAME, 'zip', EXPORT_DIR)
    
    print("HOÀN TẤT! Pipeline chạy thành công.")
    return f"{ZIP_FILENAME}.zip"

if __name__ == "__main__":
    # Test với file PDF của bạn
    test_path = r"data\TOÁN 4 - PHIẾU 1.pdf" 
    try:
        zip_file = main_pipeline(test_path)
        print(f"Sản phẩm đầu ra sẵn sàng tại: {os.path.abspath(zip_file)}")
    except Exception as e:
        print(f" Lỗi hệ thống: {e}")
