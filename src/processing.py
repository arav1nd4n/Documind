import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

class DocumentPreprocessor:
    def __init__(self, dpi=200):
        self.dpi = dpi
        
    def pdf_to_image(self, pdf_path):
        """Convert PDF to high-quality image"""
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    def enhance_image(self, image):
        """Improve image quality for OCR"""
        img_array = np.array(image)
        
        # Contrast enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        
        # Sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(limg, -1, kernel)
        
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_LAB2RGB))
    
    def preprocess(self, pdf_path):
        """Full preprocessing pipeline"""
        raw_image = self.pdf_to_image(pdf_path)
        return self.enhance_image(raw_image)# Document preprocessing logic
