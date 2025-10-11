# test_marian_ko2en.py
# -*- coding: utf-8 -*-
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-ko-en"  # ko->en는 non-tc-big이 더 안정적임
tok = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

src = "와, 또 그걸 해냈네. 정말 대단하다."
enc = tok([src], return_tensors="pt", padding=True, truncation=True)
out = model.generate(
    **enc,
    do_sample=False,       # 샘플링 OFF
    num_beams=5,           # 빔서치
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.decode(out[0], skip_special_tokens=True))
