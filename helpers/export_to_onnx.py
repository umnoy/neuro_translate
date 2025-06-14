from transformers import MarianMTModel
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.config import OnnxConfig
from pathlib import Path

model_id = "Helsinki-NLP/opus-mt-ru-en"
output_dir = Path("../models/opus-mt-ru-en")

main_export(
    model_name_or_path=model_id,
    output=output_dir,
    task="translation",
    no_post_process=True,
)
