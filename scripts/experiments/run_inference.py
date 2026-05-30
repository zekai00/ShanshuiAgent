import json
import torch
import re  # <--- 加上了这个！
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. 路径与参数配置
# ==========================================
base_model_path = "/root/models/Qwen3.5-9B"
lora_path = "/root/Workspace/LLaMA-Factory/saves/Qwen3.5-9B-Base/lora/train_landscape_sft_v2" 
test_file = "/root/Workspace/ChineseLandscape/data/auto_generated_queries.json"
output_file = "/root/Workspace/ChineseLandscape/data/lora_answers.json" # 输出为 LoRA 答案

# ==========================================
# 2. 加载基座模型与 Tokenizer
# ==========================================
print("[*] 正在加载基座模型入显存 (全量塞入 GPU 0)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cuda:0", # 🌟 锁死单卡，彻底告别 PCIe 跨卡通信瓶颈！
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# ==========================================
# 3. 挂载并物理焊死 LoRA 补丁
# ==========================================
print(f"[*] 正在挂载专属 LoRA 补丁: {lora_path}")
model = PeftModel.from_pretrained(base_model, lora_path)

print("[*] ⚡ 正在执行 merge_and_unload，合并矩阵权重 (约需几十秒)...")
model = model.merge_and_unload() # 🌟 解除 DoRA 动态计算惩罚
model.eval()

# ==========================================
# 4. 读取问题并开始推理
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
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # 🌟 放大 Token 限制，允许它充分思考
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=2048, 
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 提取新生成的文本
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 🌟 后处理：剥离思考过程标签
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # Fallback 保底机制
        if "thinking process" in response or "思考过程" in response:
            parts = response.split('\n\n')
            if len(parts) > 1:
                response = '\n\n'.join(parts[1:]).strip()

    answers.append(response)

# ==========================================
# 5. 保存结果
# ==========================================
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)

print(f"\n🎉 专属山水画大模型 (LoRA) 解答完毕！已保存至 {output_file}")