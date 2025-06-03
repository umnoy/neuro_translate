from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.save_pretrained("onnx/tokenizer")
