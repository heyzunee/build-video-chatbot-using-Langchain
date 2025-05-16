import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Predict:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

    def predict(self, history):
        input_text = self.tokenizer.apply_chat_template(history, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs, max_new_tokens=100, temperature=0.2, top_p=0.9, do_sample=True
        )
        decoded = self.tokenizer.decode(outputs[0])
        response = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        return response
