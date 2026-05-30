import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 路径与参数配置
# ==========================================
base_model_path = "/root/models/Qwen3.5-9B"
test_file = "/root/Workspace/ChineseLandscape/data/auto_generated_queries.json"
output_file = "/root/Workspace/ChineseLandscape/data/base_answers.json" # 输出为 base 答案

# ==========================================
# 2. 加载基座模型与 Tokenizer (无 LoRA)
# ==========================================
print("[*] 正在加载纯净版基座模型入显存...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cuda:1",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
base_model.eval()

# ==========================================
# 3. 读取问题并开始推理
# ==========================================
with open(test_file, 'r', encoding='utf-8') as f:
    queries = json.load(f)

print(f"[*] 开始解答 {len(queries)} 个问题...")
answers = []

for i, query in enumerate(queries):
    print(f"  -> 正在解答 [{i+1}/{len(queries)}]")
    
    # 🌟 强化 Prompt：死死按住它，只准用中文，且必须给出明确回答
    messages = [
        {
            "role": "system", 
            "content": "你是一个专业的中国传统山水画学者。请务必全程使用【中文】进行思考和解答，绝对禁止输出英文。"
        },
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(base_model.device)
    
    with torch.no_grad():
        # 🌟 放大 Token 限制，允许它充分思考
        generated_ids = base_model.generate(
            **inputs, 
            max_new_tokens=2048,  # 从 512 提升到 2048
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 提取新生成的文本
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 🌟 后处理：如果它还是输出了包含 <think> 的标签，或者是英文的 "Here's a thinking process..."，我们尝试剥离它
        # 1. 尝试剥离标准的 <think> 标签
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # 2. 如果是用纯文本写的思考过程，我们尝试以双换行符分割，取最后一段作为正式回答
        if "thinking process" in response or "思考过程" in response:
            parts = response.split('\n\n')
            if len(parts) > 1:
                # 抛弃第一段思考，保留后面的所有回答
                response = '\n\n'.join(parts[1:]).strip()

    answers.append(response)

# ==========================================
# 4. 保存结果
# ==========================================
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)

print(f"\n🎉 原版基座模型解答完毕！已保存至 {output_file}")