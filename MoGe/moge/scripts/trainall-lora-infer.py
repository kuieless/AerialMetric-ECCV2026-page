import torch
import cv2
import numpy as np
import json
import os
import argparse
import sys
import matplotlib
matplotlib.use('Agg') # 防止无界面报错
import matplotlib.pyplot as plt
from pathlib import Path
from peft import LoraConfig, get_peft_model

# ================= 🔧 环境路径设置 =================
# 自动向上寻找 moge 包的位置，确保能 import moge
current_file = Path(__file__).resolve()
project_root = str(current_file.parents[2]) # 根据你的目录层级调整，通常是指向 MoGe/ 根目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from moge.model import import_model_class_by_version
from moge.utils.vis import colorize_depth, colorize_normal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config.json used in training")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained .pt file")
    parser.add_argument('--input', type=str, required=True, help="Path to input image")
    parser.add_argument('--output_dir', type=str, default='results_inference')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inference device: {device}")

    # ================= 1. 初始化 Base Model =================
    print(f"📂 Loading Config: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    print("🏗️  Initializing Base Model (Original MoGe)...")
    MoGeModel = import_model_class_by_version(config['model_version'])
    model = MoGeModel(**config['model'])
    
    # ================= 2. 施加 LoRA (关键！必须与训练一致) =================
    print("✨ Applying PEFT LoRA...")
    
    # ⚠️⚠️⚠️ 请检查这里！必须和你最后一次成功训练的 train.py 里的配置完全一致 ⚠️⚠️⚠️
    # 如果你采用了“只学尺度”策略，HEADS_TO_SAVE 应该只包含 ["scale_head"]
    # 如果你是全量 Head 训练，则保留所有。
    # 这里默认写全，如果你训练时去掉了 points_head，请务必在这里删掉它！
    
    LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]
    
    # [请根据你的实际训练代码修改这个列表]
    # HEADS_TO_SAVE = ["points_head", "mask_head", "normal_head", "scale_head"] 
    HEADS_TO_SAVE = ["scale_head"] # <--- 如果你用了保细节策略，用这个
    
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        bias="none",
        target_modules=LORA_TARGETS,
        modules_to_save=HEADS_TO_SAVE 
    )
    
    model = get_peft_model(model, peft_config)
    model.to(device)
    model.eval()

    # ================= 3. 智能加载训练权重 =================
    print(f"📦 Loading Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # --- 复用训练时的智能匹配逻辑 ---
    print("🔧 Matching keys...")
    new_state_dict = {}
    model_keys = set(model.state_dict().keys())
    
    for k, v in state_dict.items():
        # 1. 尝试直接匹配 (最理想情况)
        if k in model_keys:
            new_state_dict[k] = v
            continue
            
        # 2. 尝试加上 base_model.model. 前缀 (常见情况)
        prefixed_k = f"base_model.model.{k}"
        if prefixed_k in model_keys:
            new_state_dict[prefixed_k] = v
            continue
            
        # 3. 尝试处理 LoRA base_layer (如果你加载的是旧的 checkpoint)
        parts = prefixed_k.split('.')
        if parts[-1] in ['weight', 'bias']:
            base_injected_k = ".".join(parts[:-1] + ["base_layer", parts[-1]])
            if base_injected_k in model_keys:
                new_state_dict[base_injected_k] = v
                continue
        
        # 4. 尝试处理 Modules to Save (Head)
        # 如果 checkpoint 里是 points_head.xxx，但模型里是 ...points_head.modules_to_save.default.xxx
        for head in HEADS_TO_SAVE:
            if k.startswith(head):
                suffix = k[len(head)+1:]
                trainable_k = f"base_model.model.{head}.modules_to_save.default.{suffix}"
                if trainable_k in model_keys:
                    new_state_dict[trainable_k] = v
                    break

    # 执行加载
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Weights Loaded.")
    
    # 检查 LoRA 是否加载成功
    lora_keys = [k for k in new_state_dict.keys() if "lora" in k]
    if len(lora_keys) > 0:
        print(f"🎉 Success: {len(lora_keys)} LoRA params loaded.")
    else:
        print("⚠️ Warning: No LoRA params found in checkpoint. Check your loading logic if this is unexpected.")

    # ================= 4. 推理过程 =================
    print(f"🖼️  Processing image: {args.input}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取图像
    img_raw = cv2.imread(args.input)
    if img_raw is None:
        raise ValueError(f"Could not open image: {args.input}")
    
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    
    # 预处理: 归一化 + 转 Tensor
    # MoGe 的 infer() 接口非常智能，会自动处理 resize 和 padding，所以这里只要转成 Tensor 即可
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device) # [1, 3, H, W]
    
    with torch.no_grad():
        # 调用 infer 接口
        if hasattr(model, 'infer'):
            output = model.infer(img_tensor)
        else:
            # 穿透 PEFT 包装
            output = model.base_model.model.infer(img_tensor)

    # ================= 5. 解析与保存结果 =================
    # 获取输出
    pred_depth = output['depth'].squeeze().cpu().numpy()   # [H, W]
    pred_mask = output['mask'].squeeze().cpu().numpy() if 'mask' in output else None
    pred_points = output['points'].squeeze().cpu().numpy() # [H, W, 3]
    pred_normal = output['normal'].squeeze().cpu().numpy() if 'normal' in output else None
    
    stem = Path(args.input).stem
    
    # --- 1. 保存彩色深度图 (带范围文字) ---
    vis_depth = colorize_depth(pred_depth, pred_mask)
    
    # 计算有效范围
    valid_vals = pred_depth[pred_mask > 0] if pred_mask is not None else pred_depth
    vmin, vmax = valid_vals.min(), valid_vals.max()
    
    # 在图上写字
    info_text = f"Depth: {vmin:.2f}m - {vmax:.2f}m"
    cv2.putText(vis_depth, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4) # 黑边
    cv2.putText(vis_depth, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2) # 白字
    
    vis_path = os.path.join(args.output_dir, f"{stem}_vis.jpg")
    cv2.imwrite(vis_path, cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR))
    
    # --- 2. 保存带 Colorbar 的图表 (Matplotlib) ---
    try:
        plt.figure(figsize=(10, 8))
        d_plot = pred_depth.copy()
        if pred_mask is not None: d_plot[pred_mask == 0] = np.nan
        
        # 使用 Robust Range 防止噪点破坏色阶
        r_min, r_max = np.percentile(valid_vals, 1), np.percentile(valid_vals, 99)
        
        plt.imshow(d_plot, cmap='Spectral_r', vmin=r_min, vmax=r_max)
        cbar = plt.colorbar()
        cbar.set_label('Metric Depth (m)')
        plt.title(f'Depth Prediction: {stem}')
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, f"{stem}_chart.png"), bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        print(f"Chart plotting failed: {e}")

    # --- 3. 保存原始数据 ---
    # 保存为 16位 PNG (单位毫米)
    depth_uint16 = (pred_depth * 1000).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(args.output_dir, f"{stem}_raw.png"), depth_uint16)
    
    # 保存点云 .npy
    np.save(os.path.join(args.output_dir, f"{stem}_points.npy"), pred_points)

    print(f"🎉 Inference Done! Results saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()


'''
python /home/szq/moge2/MoGe/moge/scripts/trainall-lora-infer.py \
  --config /home/szq/moge2/MoGe/configs/Final_train/train-l-patch-lora.json \
  --checkpoint /home/szq/moge2/MoGe/workspace/final-fintune2-130-lora-final3/checkpoint/00000000.pt \
  --input /home/szq/moge2/MoGe/workspace/final-fintune2-130-lora-final3/vis/step_00000000/0026/image.jpg \
  --output_dir my_results




'''