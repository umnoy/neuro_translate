from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

def translate(text, src_lang_code, tgt_lang_code):
    tokenizer.src_lang = src_lang_code
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code])
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Примеры:
ru_to_en = translate("У меня в жопе черви", "ru_RU", "en_XX")
en_to_ru = translate("I have some worms in my ass", "en_XX", "ru_RU")

print("RU → EN:", ru_to_en)
print("EN → RU:", en_to_ru)
