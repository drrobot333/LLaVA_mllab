import torch
import argparse  # 추가
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저 로드
MODEL_PATH = "/workspace/hdd/Qwen2.5-72B-Instruct-AWQ"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # 모델 크기에 맞춰 float16 사용
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def generate_response(prompt: str, max_tokens: int = 512):
    """주어진 프롬프트에 대해 모델 응답 생성"""
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )
    
    # 입력 부분 이후 생성된 토큰만 추출
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# Flask 앱 생성
app = Flask(__name__)

@app.route('/send', methods=['POST'])
def send():
    """클라이언트로부터 프롬프트를 받아 모델 응답을 생성"""
    data = request.json
    prompt = data.get("text", "")
    max_tokens = data.get("max_tokens", 512)
    
    if not prompt:
        return jsonify({"error": "Text is required"}), 400
    
    response = generate_response(prompt, max_tokens)
    return jsonify({"response": response})

if __name__ == '__main__':
    # argparse를 사용하여 포트를 지정할 수 있도록 수정
    parser = argparse.ArgumentParser(description="Qwen Flask Server")
    parser.add_argument('-p', '--port', type=int, default=6000, help="서버 포트 (기본값: 6000)")
    args = parser.parse_args()

    # 지정된 포트에서 Flask 실행
    app.run(host="0.0.0.0", port=args.port)
