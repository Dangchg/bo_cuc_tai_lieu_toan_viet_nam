import cv2
from PIL import Image
import numpy as np
from paddleocr import PPStructureV3
from paddleocr import PaddleOCRVL
import os
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.9" #tăng lên 1 nếu Vram của gpu lớn 
# Khởi tạo các model toàn cục
pipeline = PaddleOCRVL(
                       use_doc_orientation_classify = False,
                        use_doc_unwarping = False,
                        use_layout_detection = False,
                        use_chart_recognition = False,
                        use_seal_recognition= False,
                        use_ocr_for_image_block=False,
                        use_queues=False,)

layout_model = PPStructureV3(lang="vi",
                      use_seal_recognition=False,
                        use_chart_recognition=False,
                        use_formula_recognition=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)

def predict_vietocr(image_crop):
    img_array = np.array(image_crop)
    text = pipeline.predict(img_array)
    contents = []
    for item in text:
        for block in item["parsing_res_list"]:
            contents.append(block.content)
    return "\n".join(contents)

def predict_math_latex(image_crop):
    img_array = np.array(image_crop)
    output = pipeline.predict(img_array)
    contents = []
    for item in output:
        for block in item["parsing_res_list"]:
            contents.append(block.content)
    return "\n".join(contents)

"""def get_layout_regions(image_path):
    output = layout_model.predict(image_path,layout_nms=True,use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)
    regions = []
    for box in output[0]['boxes']:
        regions.append({'label': box['label'], 'box': [int(c) for c in box['coordinate']]})
    return regions"""

def get_layout_regions(image_path):
    output = layout_model.predict(image_path)
    all_regions = []
    
    for page in list(output):
        title_boxes = []
        
        # 1. Lấy Layout (Image, Formula, Table, Chart) VÀ thu thập Title
        for box in page.get("layout_det_res", {}).get("boxes", []):
            # Dùng strip() để cắt khoảng trắng thừa, tránh lỗi nhãn 'chart ' không khớp 'chart'
            label = str(box['label']).lower().strip() 
            score = float(box.get('score', 1.0))
            coords = [int(c) for c in box['coordinate']]
            print(f"[AI RAW DETECT] Thấy 1 khối: {label} (Độ tự tin: {score})")
            
            # KIỂM TRA: Gom đầy đủ các nhãn hình ảnh và toán học
            if label in ['image', 'chart', 'figure', 'formula', 'texformula']:
                all_regions.append({'label': label, 'box': coords, 'score': score})
                
            # Thu thập các box tiêu đề
            elif label in ['title', 'doc_title', 'paragraph_title']:
                title_boxes.append({'label': label, 'box': coords, 'score': score})
                
        # 2. Xử lý Bảng 
        if "table_res_list" in page:
            for table in page["table_res_list"]:
                if "cell_box_list" in table and len(table["cell_box_list"]) > 0:
                    cell_boxes = []
                    min_x, min_y = float('inf'), float('inf')
                    max_x, max_y = 0, 0
                    for cb in table["cell_box_list"]:
                        x1, y1, x2, y2 = [int(x) for x in cb]
                        cell_boxes.append([x1, y1, x2, y2])
                        min_x, min_y = min(min_x, x1), min(min_y, y1)
                        max_x, max_y = max(max_x, x2), max(max_y, y2)
                    all_regions.append({
                        'label': 'table',
                        'box': [min_x, min_y, max_x, max_y], 
                        'cell_box_list': cell_boxes,
                        'score': 1.0  # THÊM ĐIỂM SỐ CHO BẢNG ĐỂ KHÔNG BỊ XÓA OAN
                    })
                
        # 3. Lấy Line Box và KIỂM TRA CHÉO VỚI TITLE BOXES
        if "overall_ocr_res" in page:
            rec_boxes = page["overall_ocr_res"].get("rec_boxes", [])
            rec_res = page["overall_ocr_res"].get("rec_res", []) # Chứa điểm số thật của text
            
            for idx, box in enumerate(rec_boxes):
                line_coords = [int(x) for x in box]
                line_label = 'text_line'
                
                # BẮT BUỘC: Reset lại điểm score cho text_line (Vá lỗi rò rỉ biến)
                line_score = 1.0
                if idx < len(rec_res) and isinstance(rec_res[idx], (list, tuple)) and len(rec_res[idx]) >= 2:
                    try:
                        line_score = float(rec_res[idx][1])
                    except:
                        pass
                
                # Tính tọa độ tâm (center) của dòng chữ
                cy = (line_coords[1] + line_coords[3]) / 2
                cx = (line_coords[0] + line_coords[2]) / 2
                
                # So khớp hình học: Tâm dòng chữ nằm trong box Title nào thì lấy nhãn của box đó
                for t_box in title_boxes:
                    tx1, ty1, tx2, ty2 = t_box['box']
                    if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                        line_label = t_box['label'] 
                        break
                        
                all_regions.append({
                    'label': line_label,
                    'box': line_coords,
                    'score': line_score # DÙNG ĐÚNG ĐIỂM SỐ CỦA DÒNG CHỮ
                })
                
    return all_regions
