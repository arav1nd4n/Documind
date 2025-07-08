from prefect import flow, task
import os
from .processing import DocumentPreprocessor
from .extraction import AIExtractor
from .validation import ValidationEngine
import json

@task(retries=2, retry_delay_seconds=10)
def process_document(file_path, output_dir="results"):
    """End-to-end processing of a single document"""
    # Initialize components
    preprocessor = DocumentPreprocessor()
    extractor = AIExtractor()
    validator = ValidationEngine()
    
    # Process document
    image = preprocessor.preprocess(file_path)
    extracted_data = extractor.extract_entities(image)
    validation = validator.validate(extracted_data)
    
    # Save results
    filename = os.path.basename(file_path)
    result = {
        "document": filename,
        "extracted_data": extracted_data,
        "validation": validation,
        "processing_time": datetime.now().isoformat()
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
        
    return result

@flow(name="documind_pipeline")
def main_flow(directory="data/invoices"):
    """Process all documents in a directory"""
    results = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            results.append(process_document(file_path))
            
    return results# Workflow orchestration logic
