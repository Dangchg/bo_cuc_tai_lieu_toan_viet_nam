import numpy as np

def calculate_ioa(box1, box2):
    """Tính phần trăm diện tích box1 nằm bên trong box2 (Intersection over Area)"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    if box1_area == 0: return 0.0
    return inter_area / box1_area

def assign_lines_to_cells(target_table_box, cell_box_list, all_regions):
    """
    Hút các text_line/formula vào đúng cell_box tương ứng.
    Trả về: (Dữ liệu các Cell, Các regions không thuộc bảng này)
    """
    table_items = []
    outside_items = []
    
    # 1. Lọc các đối tượng nằm trong bảng
    for item in all_regions:
        # --- CHỐT CHẶN QUAN TRỌNG NHẤT ---
        # Nếu đây chính là cái bảng đang được xử lý -> KHÔNG thêm vào outside_items nữa
        # Điều này giúp vòng lặp while ở main.py không bị kẹt vô hạn
        if item['label'] == 'table' and item['box'] == target_table_box:
            continue
            
        # Các nhãn image, figure hoặc CÁC BẢNG KHÁC thì vẫn giữ lại cho vòng lặp sau
        if item['label'] in ['table', 'image', 'chart', 'figure']:
            outside_items.append(item)
            continue
            
        # Nếu text_line nằm > 50% trong bảng -> Nó thuộc về bảng
        if calculate_ioa(item['box'], target_table_box) > 0.5:
            table_items.append(item)
        else:
            outside_items.append(item)

    # 2. Khởi tạo các thùng chứa cho từng Cell
    cells_data = {i: {'cell_box': cell_box_list[i], 'contents': []} for i in range(len(cell_box_list))}
    
    # 3. Gán từng dòng chữ vào Cell phù hợp nhất
    for item in table_items:
        best_cell_idx = -1
        max_ioa = 0
        
        for i, cell in enumerate(cell_box_list):
            ioa = calculate_ioa(item['box'], cell)
            if ioa > max_ioa:
                max_ioa = ioa
                best_cell_idx = i
                
        # Nếu dòng chữ có độ giao cắt > 30% với 1 cell, ta đưa nó vào cell đó
        if best_cell_idx != -1 and max_ioa > 0.3:
            cells_data[best_cell_idx]['contents'].append(item)
            
    # 4. Sắp xếp lại thứ tự chữ TRONG CÙNG 1 CELL (Ép sai số trục Y là 15 pixel)
    for i in cells_data:
        cells_data[i]['contents'].sort(key=lambda x: (x['box'][1] // 15, x['box'][0]))
        
    return cells_data, outside_items

def build_latex_table(cells_text_list, cell_box_list):
    """
    Tái tạo lại cấu trúc Hàng/Cột từ danh sách tọa độ Cell phẳng.
    Tự động chia tỷ lệ cột để văn bản xuống dòng, không bị tràn khổ giấy.
    """
    if not cell_box_list: return ""
    
    cells = [{'box': cell_box_list[i], 'text': cells_text_list[i]} for i in range(len(cell_box_list))]
    cells.sort(key=lambda c: c['box'][1])
    
    rows = []
    current_row = [cells[0]]
    
    for c in cells[1:]:
        base_y = current_row[0]['box'][1]
        tolerance = (current_row[0]['box'][3] - current_row[0]['box'][1]) * 0.5
        
        if abs(c['box'][1] - base_y) < tolerance:
            current_row.append(c)
        else:
            current_row.sort(key=lambda x: x['box'][0])
            rows.append(current_row)
            current_row = [c]
            
    if current_row:
        current_row.sort(key=lambda x: x['box'][0])
        rows.append(current_row)
        
    max_cols = max([len(r) for r in rows]) if rows else 0
    if max_cols == 0: return ""
    
    # TÍNH TOÁN ĐỘ RỘNG CỘT TỰ ĐỘNG
    # Lấy 95% chiều rộng trang giấy chia đều cho số cột (chừa 5% cho đường kẻ và padding)
    col_fraction = round(0.95 / max_cols, 3)
    
    # Đổi từ cột 'c' sang cột 'p' (paragraph) 
    col_format = f"p{{{col_fraction}\\textwidth}}|" * max_cols
    
    latex_str = "\\begin{center}\n\\begin{tabular}{|" + col_format + "}\n\\hline\n"
    
    for row in rows:
        row_texts = [c['text'] for c in row]
        while len(row_texts) < max_cols:
            row_texts.append("")
        
        latex_str += " & ".join(row_texts) + " \\\\\n\\hline\n"
        
    latex_str += "\\end{tabular}\n\\end{center}"
    return latex_str
