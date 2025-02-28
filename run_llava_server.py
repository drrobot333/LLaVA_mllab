import io
import re
import torch
from flask import Flask, request
from PIL import Image

import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

# 모델과 관련 설정 (실제 경로 및 기본값으로 수정)
MODEL_PATH = "/workspace/hdd/llava-v1.5-13b"   # 모델 파일 경로
MODEL_BASE = None                 # 모델 베이스 이름 (필요시)

# 모델 초기화 (서버가 시작될 때 한 번만 로드)
disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(MODEL_PATH, MODEL_BASE, model_name)
device = model.device  # 모델이 할당된 디바이스

# Flask 앱 생성
app = Flask(__name__)

@app.route('/send_image_text', methods=['POST'])
def send_image_text():
    """
    클라이언트로부터 'text' (프롬프트)와 'image' 파일을 받아 모델 추론을 수행한 후 결과 텍스트를 반환합니다.
    """
    # 클라이언트 요청에서 텍스트와 이미지 파일 읽기
    text = request.form.get('text', '')
    image_file = request.files.get('image', None)
    if image_file is None:
        return "No image provided", 400

    try:
        image = Image.open(image_file.stream).convert('RGB')
    except Exception as e:
        return f"Error processing image: {str(e)}", 400

    # 이미지 토큰 생성 및 텍스트에 추가 (eval_model 코드 참고)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    qs = text
    if IMAGE_PLACEHOLDER in qs:
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # 모델 이름에 따른 대화 모드 결정
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # 대화 템플릿 생성 (eval_model의 conv_templates 활용)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 이미지 전처리: 이미지 크기 기록 및 텐서 변환
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)

    # 토크나이저를 이용해 프롬프트를 토큰화
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)

    # 추론 수행
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if 0.7 > 0 else False,  # 예시로 temperature=0.7 사용
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

if __name__ == '__main__':
    # 기본 포트 7000번에서 서버 실행 (클라이언트 코드와 맞춰야 함)
    app.run(port=7000)
