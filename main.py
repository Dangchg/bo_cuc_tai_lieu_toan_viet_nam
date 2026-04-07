import cv2
import os
import shutil
import fitz  # PyMuPDF
import numpy as np

from models import get_layout_regions, predict_vietocr, predict_math_latex
from image_utils import smart_crop, draw_boxes_on_image, process_layout_boxes, enhance_image, read_pdf_to_cv2_images

# ==========================================
# CẤU HÌNH
# ==========================================
DEBUG_MODE = True
DEBUG_DIR = "./debug_results"
EXPORT_DIR = "./ket_qua_xuat"
ZIP_FILENAME = "Tai_Lieu_So_Hoa"

def export_to_latex(content_list, output_filename="output.tex"):
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
    body_content = "\n\n".join(content_list)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(latex_header)
        f.write(body_content)
        f.write("\n" + latex_footer)
    print(f"-> Đã xuất file thành công: {output_filename}")


def process_single_image(img_goc, page_index=1):
    """Xử lý đơn lẻ cho 1 trang ảnh và xuất debug theo trang."""
    results = []
    
    # 1. Lưu file tạm để PPStructure đọc
    temp_img_path = f"temp_page_{page_index}.jpg"
    cv2.imwrite(temp_img_path, img_goc)
    
    print(f"--- Đang xử lý trang {page_index} ---")
    raw_regions = get_layout_regions(temp_img_path)
    
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    # === DEBUG 1: RAW LAYOUT ===
    if DEBUG_MODE:
        draw_boxes_on_image(img_goc, raw_regions, os.path.join(DEBUG_DIR, f"1_raw_layout_p{page_index}.jpg"))

    # 2. Xử lý hình học
    ordered_boxes = process_layout_boxes(img_goc, raw_regions)
    
    # === DEBUG 2: ORDERED LAYOUT ===
    if DEBUG_MODE:
        draw_boxes_on_image(img_goc, ordered_boxes, os.path.join(DEBUG_DIR, f"2_final_order_p{page_index}.jpg"), draw_order=True)

    # 3. Phân luồng cắt ảnh và OCR
    for i, item in enumerate(ordered_boxes):
        x1, y1, x2, y2 = item['box']
        label = item['label']
        
        # --- XỬ LÝ ẢNH MINH HỌA ---
        if label in ['image', 'figure']:
            pad = 5
            c_y1, c_y2 = max(0, y1 - pad), min(img_goc.shape[0], y2 + pad)
            c_x1, c_x2 = max(0, x1 - pad), min(img_goc.shape[1], x2 + pad)
            figure_crop = img_goc[c_y1:c_y2, c_x1:c_x2]
            
            if figure_crop.size == 0: continue
            
            # === DEBUG 3: LƯU CROP ẢNH MINH HỌA ===
            if DEBUG_MODE:
                cv2.imwrite(os.path.join(DEBUG_DIR, "crops", f"part_p{page_index}_{i+1:03d}_image.jpg"), figure_crop)
            
            img_name = f"hinh_minh_hoa_p{page_index}_{i+1}.jpg"
            cv2.imwrite(os.path.join(EXPORT_DIR, img_name), figure_crop)
            
            latex_img_code = f"\\includegraphics[max width=0.8\\textwidth, center]{{{img_name}}}"
            results.append(latex_img_code)
            continue

        # --- XỬ LÝ TEXT VÀ MATH ---
        expanded_box = [x1, y1, x2, y2]
        crop = smart_crop(img_goc, expanded_box, pad_ratio=0.05)
        
        # Đã bắt kỹ trường hợp None và size rỗng
        if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5: 
            continue
            
        # === DEBUG 3: LƯU CROP CHỮ/TOÁN (TRƯỚC KHI LÀM NÉT) ===
        if DEBUG_MODE:
            cv2.imwrite(os.path.join(DEBUG_DIR, "crops", f"part_p{page_index}_{i+1:03d}_{label}.jpg"), crop)

        enhanced_crop = enhance_image(crop)
        
            
        
        if label in ['text', 'text_line', 'paragraph_title', 'doc_title']:
            text_res = predict_vietocr(enhanced_crop)
            results.append(text_res)
            
        elif label in ['formula', 'texformula']:
            math_res = predict_math_latex(enhanced_crop)
            results.append(math_res)
            
    return results


def main_pipeline(input_path):
    # 0. Chuẩn bị thư mục Debug và Export
    if DEBUG_MODE:
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(os.path.join(DEBUG_DIR, "crops"))
        
    if os.path.exists(EXPORT_DIR): shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR)
    
    # 1. Đọc dữ liệu đầu vào (PDF hoặc Ảnh)
    images_to_process = []
    
    if input_path.lower().endswith('.pdf'):
        images_to_process = read_pdf_to_cv2_images(input_path, dpi=300)
    else:
        img_cv = cv2.imread(input_path)
        if img_cv is None:
            raise ValueError(f"Không thể đọc ảnh tại: {input_path}")
        images_to_process.append(img_cv)
        
    all_results = []
    
    # 2. Xử lý qua từng trang
    for page_idx, img in enumerate(images_to_process):
        page_results = process_single_image(img, page_index=page_idx+1)
        all_results.extend(page_results)
        
        # Thêm ngắt trang trong LaTeX giữa các trang PDF
        all_results.append(r"\newpage")
        
    # 3. Xuất file và Nén ZIP
    print("-> Đang tạo file LaTeX tổng...")
    tex_filepath = os.path.join(EXPORT_DIR, "ket_qua_bai_toan.tex")
    
    if all_results and all_results[-1] == r"\newpage":
        all_results.pop()
        
    export_to_latex(all_results, tex_filepath)
    
    print(f"-> Đang nén dữ liệu thành {ZIP_FILENAME}.zip ...")
    shutil.make_archive(ZIP_FILENAME, 'zip', EXPORT_DIR)
    
    print("HOÀN TẤT! Toàn bộ kết quả đã được đóng gói.")
    return f"{ZIP_FILENAME}.zip"

if __name__ == "__main__":
    test_path = r"data\dd.png" 
    try:
        zip_file = main_pipeline(test_path)
        print(f"File đầu ra sẵn sàng tại: {os.path.abspath(zip_file)}")
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")