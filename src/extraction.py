from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import re

class AIExtractor:
    def __init__(self, model_name="naver-clova-ix/donut-base-finetuned-cord-v2"):
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.eval()
        
    def extract_entities(self, image):
        """Extract structured data from document images"""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Generate output
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            return_dict_in_generate=True,
        )
        
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
            self.processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        return self.processor.token2json(sequence)# AI entity extraction logic
