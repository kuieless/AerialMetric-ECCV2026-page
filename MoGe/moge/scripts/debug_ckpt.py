import torch
import json
from pathlib import Path
import os
import sys
# ================= 🔧 修复路径问题 =================
# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()
# 向上找 3 层，找到项目根目录 (即 .../MoGe/)
project_root = str(current_file.parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"🔧 Added to sys.path: {project_root}")
from pathlib import Path
from peft import LoraConfig, get_peft_model
from moge.model import import_model_class_by_version


# ================= 配置区域 =================
config_path = "/home/szq/moge2/MoGe/configs/Final_train/train-l-patch.json"
checkpoint_path = "/home/szq/moge2/MoGe/vitl-normal.pt"
# ===========================================

def main():
    print(f"📂 Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("🏗️  Initializing Model...")
    MoGeModel = import_model_class_by_version(config['model_version'])      
    model = MoGeModel(**config['model'])

    print("✨ Wrapping with PEFT (LoRA)...")
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["qkv", "proj", "fc1", "fc2"],
        modules_to_save=["points_head", "mask_head", "normal_head", "scale_head"]
    )
    model = get_peft_model(model, peft_config)
    
    # 获取模型当前期望的所有 Key
    model_keys = set(model.state_dict().keys())
    print(f"🤖 Model expects {len(model_keys)} keys.")
    
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    print(f"💾 Checkpoint has {len(ckpt_state_dict)} keys.")

    print("\n🔍 --- DIAGNOSIS: Key Mismatch Analysis ---")
    # 打印几个样本看看区别
    print("Example Model Key:     ", list(model_keys)[0])
    print("Example Checkpoint Key:", list(ckpt_state_dict.keys())[0])
    
    print("\n🛠️  --- ATTEMPTING AUTO-MAPPING ---")
    new_state_dict = {}
    
    # 定义需要全量保存的 Head 名字
    heads = ["points_head", "mask_head", "normal_head", "scale_head"]
    
    # 计数器
    mapped_count = 0
    lora_injected_count = 0
    head_duplicated_count = 0
    
    for k, v in ckpt_state_dict.items():
        # 1. 基础映射：直接加 PEFT 前缀
        # 原始: encoder.xxx -> base_model.model.encoder.xxx
        # 原始: neck.xxx    -> base_model.model.neck.xxx
        if k.startswith("encoder") or k.startswith("neck"):
            # 检查是否命中了 LoRA 层 (需要注入 base_layer)
            # 这里的判断逻辑是：如果模型里有这个 key 加上 .base_layer，那它就是 LoRA 层
            target_key_normal = f"base_model.model.{k}"
            
            # 分割 key 来插入 base_layer (针对 .weight 和 .bias)
            parts = target_key_normal.split('.')
            if parts[-1] in ['weight', 'bias']:
                key_with_base_layer = ".".join(parts[:-1] + ["base_layer", parts[-1]])
            else:
                key_with_base_layer = target_key_normal # 不太可能发生

            if key_with_base_layer in model_keys:
                new_state_dict[key_with_base_layer] = v
                lora_injected_count += 1
                # print(f"  [LoRA] {k} -> {key_with_base_layer}")
            else:
                # 普通层
                new_state_dict[target_key_normal] = v
                mapped_count += 1
        
        # 2. 处理 Heads (modules_to_save)
        # 它们比较特殊，会被 PEFT 复制成两份：一份叫 modules_to_save.default (训练用)，一份叫 original_module (冻结用)
        elif any(k.startswith(h) for h in heads):
            head_name = next(h for h in heads if k.startswith(h))
            suffix = k[len(head_name)+1:] # 去掉 head 名剩下的部分
            
            # 映射到可训练副本
            key_trainable = f"base_model.model.{head_name}.modules_to_save.default.{suffix}"
            new_state_dict[key_trainable] = v
            
            # 映射到原始副本 (消除 Missing keys 警告)
            key_original = f"base_model.model.{head_name}.original_module.{suffix}"
            new_state_dict[key_original] = v
            
            head_duplicated_count += 1
            # print(f"  [Head] {k} -> {key_trainable} & {key_original}")
            
        else:
            # 其他杂项，直接尝试加前缀或者保持原样
            target_key = f"base_model.model.{k}"
            if target_key in model_keys:
                new_state_dict[target_key] = v
            else:
                # print(f"  [Ignored] {k} (No match found)")
                pass

    print(f"\n📊 Mapping Summary:")
    print(f"  - Normal Layers Mapped: {mapped_count}")
    print(f"  - LoRA Base Layers Injected: {lora_injected_count}")
    print(f"  - Head Layers Duplicated: {head_duplicated_count}")
    
    print("\n⚖️  Loading State Dict...")
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"❌ Missing Keys: {len(msg.missing_keys)}")
    if len(msg.missing_keys) > 0:
        # 过滤掉纯 LoRA 参数 (lora_A, lora_B)，这些本来就没有
        real_missing = [k for k in msg.missing_keys if "lora_" not in k]
        print(f"❌ Real Missing Keys (excluding LoRA params): {len(real_missing)}")
        if len(real_missing) > 0:
            print("Examples:", real_missing[:5])
    else:
        print("✅ PERFECT MATCH! All base weights loaded.")

    print(f"⚠️ Unexpected Keys: {len(msg.unexpected_keys)}")

if __name__ == "__main__":
    main()