
"""
Intelligent Document Processing Pipeline
- AI-Powered Document Understanding
- Automated Validation & Fraud Detection
- Workflow Orchestration
"""
import os
import json
import re
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from datetime import datetime
from sklearn.ensemble import IsolationForest
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import cv2
import click

# ------------------------
# Configuration
# ------------------------
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_DATA_PATH = "../data/sample_invoices/"
OUTPUT_PATH = "../results/processed/"
ANOMALY_MODEL_PATH = "../models/anomaly_detector.joblib"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ------------------------
# Document Processing
# ------------------------
class DocumentProcessor:
    def __init__(self):
        self.processor = DonutProcessor.from_pretrained(MODEL_NAME)
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()
    
    def pdf_to_image(self, pdf_path, dpi=200):
        """Convert PDF to high-quality image"""
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    def preprocess_image(self, image):
        """Enhance image quality for better OCR"""
        img_array = np.array(image)
        # Apply contrast enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        limg = cv2.merge([clahe.apply(l), a, b])
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)

    def extract_document_info(self, image):
        """Use Donut model to extract structured information"""
        # Prepare inputs
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.to(DEVICE)
        
        # Generate output
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(DEVICE)
        
        # Run inference
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        
        # Process output
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
            self.processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        return self.processor.token2json(sequence)

# ------------------------
# Validation & Fraud Detection
# ------------------------
class ValidationEngine:
    def __init__(self):
        self.anomaly_model = self.train_anomaly_detector()
    
    def train_anomaly_detector(self, n_samples=1000):
        """Train fraud detection model with synthetic data"""
        # Generate normal transactions (80% of data)
        normal_totals = np.random.uniform(50, 5000, int(n_samples * 0.8))
        
        # Generate anomalies (20% of data)
        anomaly_totals = np.concatenate([
            np.random.uniform(-1000, -1, int(n_samples * 0.1)),  # Negative totals
            np.random.uniform(5001, 100000, int(n_samples * 0.1))  # Extremely high
        ])
        
        # Combine and train model
        X = np.concatenate([normal_totals, anomaly_totals]).reshape(-1, 1)
        model = IsolationForest(contamination=0.2, random_state=42)
        model.fit(X)
        return model
    
    def validate_invoice(self, invoice_data):
        """Apply business rule validation"""
        errors = []
        
        # Date validation
        if 'date' in invoice_data:
            try:
                invoice_date = datetime.strptime(invoice_data['date'], '%Y-%m-%d')
                if invoice_date > datetime.now():
                    errors.append("Future date detected")
            except ValueError:
                errors.append("Invalid date format")
        
        # Total amount validation
        if 'total' in invoice_data:
            try:
                total = float(invoice_data['total'].replace(',', ''))
                if total <= 0:
                    errors.append("Invalid total amount")
                
                # Line item validation if available
                if 'items' in invoice_data:
                    calculated_total = sum(
                        float(item['price']) * float(item['quantity']) 
                        for item in invoice_data['items']
                    )
                    if abs(calculated_total - total) > 0.01:
                        errors.append(f"Total mismatch: {total} vs {calculated_total}")
            except ValueError:
                errors.append("Invalid total format")
        
        # Fraud detection
        if 'total' in invoice_data:
            try:
                total = float(invoice_data['total'].replace(',', ''))
                is_anomaly = self.anomaly_model.predict([[total]])[0] == -1
                if is_anomaly:
                    errors.append("Potential fraud detected")
            except ValueError:
                pass
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "validation_timestamp": datetime.now().isoformat()
        }

# ------------------------
# Workflow Orchestration
# ------------------------
class ProcessingWorkflow:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.validation_engine = ValidationEngine()
    
    def process_document(self, pdf_path):
        """End-to-end document processing workflow"""
        # Step 1: Convert and preprocess document
        raw_image = self.document_processor.pdf_to_image(pdf_path)
        processed_image = self.document_processor.preprocess_image(raw_image)
        
        # Step 2: AI extraction
        extracted_data = self.document_processor.extract_document_info(processed_image)
        
        # Step 3: Validation and fraud detection
        validation_result = self.validation_engine.validate_invoice(extracted_data)
        
        # Step 4: Prepare final output
        filename = os.path.basename(pdf_path)
        result = {
            "document": filename,
            "extracted_data": extracted_data,
            "validation": validation_result,
            "processing_time": datetime.now().isoformat()
        }
        
        # Save results
        output_file = os.path.join(OUTPUT_PATH, f"{os.path.splitext(filename)[0]}.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result

# ------------------------
# CLI Interface
# ------------------------
@click.command()
@click.argument('input_path', type=click.Path(exists=True))
def process_documents(input_path):
    """Process PDF documents through AI pipeline"""
    workflow = ProcessingWorkflow()
    
    if os.path.isfile(input_path):
        # Process single file
        results = [workflow.process_document(input_path)]
    else:
        # Process directory
        results = []
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(input_path, filename)
                results.append(workflow.process_document(filepath))
    
    # Print summary
    valid_count = sum(1 for r in results if r['validation']['is_valid'])
    click.echo(f"\nProcessing Complete!")
    click.echo(f"Documents Processed: {len(results)}")
    click.echo(f"Valid Invoices: {valid_count}")
    click.echo(f"Invalid/Fraud Detected: {len(results) - valid_count}")
    click.echo(f"Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_documents()
