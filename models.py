import cv2
from PIL import Image
from paddleocr import PPStructureV3
from paddleocr import LayoutDetection
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from pix2tex.cli import LatexOCR
from modelmoi import ocrer

# Khởi tạo các model toàn cục
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cuda:0' # hoặc 'cuda:0'
vietocr_model = Predictor(config)
math_model = LatexOCR()
#layout_model = LayoutDetection(model_name="PP-DocLayout_plus-L")
layout_model = PPStructureV3(lang="vi",
                      use_seal_recognition=False,
                        use_table_recognition=False,
                        use_formula_recognition=False,
                        use_chart_recognition=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)

def predict_vietocr(image_crop):
    pil_img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    return vietocr_model.predict(pil_img).strip()

def predict_math_latex(image_crop):
    pil_img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    return f"${math_model(pil_img).strip()}$"

"""def get_layout_regions(image_path):
    output = layout_model.predict(image_path,layout_nms=True,use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)
    regions = []
    for box in output[0]['boxes']:
        regions.append({'label': box['label'], 'box': [int(c) for c in box['coordinate']]})
    return regions"""

def get_layout_regions(image_path):
    output = layout_model.predict(image_path,layout_nms=True)
    regions = []
    
    for page in list(output):
        layout_boxes = page.get("layout_det_res", {}).get("boxes", [])
        for box in layout_boxes:
            regions.append({
                'label': box['label'],
                'score': box['score'],
                'box': [int(coord) for coord in box['coordinate']] 
            })
            
    return regions




