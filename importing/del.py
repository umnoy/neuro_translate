from transformers.utils import cached_file
import shutil
import os

cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
model_dir = os.path.join(cache_dir, "models--facebook--wmt19-ru-en")

# удаление модели
shutil.rmtree(model_dir, ignore_errors=True)
