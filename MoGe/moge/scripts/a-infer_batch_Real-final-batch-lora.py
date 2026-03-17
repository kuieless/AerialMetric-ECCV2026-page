import argparse
import os
import sys
import json
import time
import logging
import traceback
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib

# 设置无头模式，防止服务器报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 🔥 引入 PEFT
from peft import LoraConfig, get_peft_model

# ================= 🔧 0. 参数配置 =================

def setup_args():
    parser = argparse.ArgumentParser(description="MoGe LoRA Batch Inference (Parallel)")
    # 🔥 新增 config 参数
    parser.add_argument("--config", type=str, required=True, help="训练使用的 config.json 路径")
    parser.add_argument("--model_path", type=str, required=True, help="训练出的 LoRA 权重路径 (.pt)")
    
    parser.add_argument("--input_root", type=str, required=True, help="数据集输入根目录")
    parser.add_argument("--output_root", type=str, required=True, help="结果输出根目录")
    parser.add_argument("--project_root", type=str, default="/home/szq/moge2/MoGe", help="MoGe项目根目录")
    
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备")
    parser.add_argument("--resize", type=int, default=None, help="缩放长边大小 (例如 1024)")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在的结果")
    
    # 🔥🔥🔥 并行分片参数 🔥🔥🔥
    parser.add_argument("--total_shards", type=int, default=1, help="并行总进程数")
    parser.add_argument("--shard_id", type=int, default=0, help="当前进程ID (0 ~ total-1)")
    
    return parser.parse_args()

# ================= 🧠 1. 推理引擎 (LoRA 核心逻辑) =================

class MoGeInferenceEngine:
    def __init__(self, config_path, checkpoint_path, device="cuda", fp16=True):
        self.device = torch.device(device)
        self.fp16 = fp16
        is_main_process = ("shard_id" not in globals() or globals().get("shard_id", 0) == 0)

        if is_main_process:
            print(f"\n📦 [系统] 正在初始化 LoRA 模型...")
            print(f"   Config: {config_path}")
            print(f"   Weights: {checkpoint_path}")

        try:
            # 1. 加载配置
            with open(config_path, 'r') as f:
                train_config = json.load(f)
            
            # 2. 初始化 Base Model
            from moge.model import import_model_class_by_version
            MoGeModel = import_model_class_by_version(train_config['model_version'])
            self.model = MoGeModel(**train_config['model'])
            
            # 3. 施加 LoRA (必须与你训练时一致)
            # 🔥 根据你的要求，只保存 scale_head
            LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]
            HEADS_TO_SAVE = ["scale_head"] 
            
            if is_main_process:
                print(f"   LoRA Targets: {LORA_TARGETS}")
                print(f"   Heads to Save: {HEADS_TO_SAVE}")

            peft_config = LoraConfig(
                r=32, 
                lora_alpha=128, 
                bias="none",
                target_modules=LORA_TARGETS,
                modules_to_save=HEADS_TO_SAVE 
            )
            self.model = get_peft_model(self.model, peft_config)
            
            # 4. 智能加载权重 (核心修复逻辑)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            new_state_dict = {}
            model_keys = set(self.model.state_dict().keys())
            
            for k, v in state_dict.items():
                # 尝试直接匹配
                if k in model_keys:
                    new_state_dict[k] = v
                    continue
                
                # 尝试加 base_model.model. 前缀
                prefixed_k = f"base_model.model.{k}"
                if prefixed_k in model_keys:
                    new_state_dict[prefixed_k] = v
                    continue
                
                # 尝试处理 LoRA base_layer
                parts = prefixed_k.split('.')
                if parts[-1] in ['weight', 'bias']:
                    base_injected_k = ".".join(parts[:-1] + ["base_layer", parts[-1]])
                    if base_injected_k in model_keys:
                        new_state_dict[base_injected_k] = v
                        continue
                
                # 尝试处理 Head (Modules to Save)
                for head in HEADS_TO_SAVE:
                    if k.startswith(head):
                        suffix = k[len(head)+1:]
                        trainable_k = f"base_model.model.{head}.modules_to_save.default.{suffix}"
                        if trainable_k in model_keys:
                            new_state_dict[trainable_k] = v
                            break

            msg = self.model.load_state_dict(new_state_dict, strict=False)
            
            if is_main_process:
                real_missing = [k for k in msg.missing_keys if "lora_" not in k]
                print(f"✅ 权重加载完成。Missing (非LoRA): {len(real_missing)}")
            
            self.model.to(self.device)
            self.model.eval()
            
            if self.fp16:
                self.model.half()
                
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            traceback.print_exc()
            sys.exit(1)

    def process_scene(self, input_dir, output_dir, config):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = sorted([p for p in input_path.rglob('*') if p.suffix.lower() in VALID_IMG_EXTS])

        if not image_files: return

        # 推理循环
        for img_path in image_files:
            rel_path = img_path.relative_to(input_path)
            save_dir = output_path / rel_path.parent / img_path.stem
            
            if not config.get('overwrite', False):
                if (save_dir / 'depth.npy').exists():
                    continue

            try:
                self._infer_single_image(img_path, save_dir, config)
            except torch.cuda.OutOfMemoryError:
                print(f"💥 [OOM] Skip: {img_path.name}")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ [Error] {img_path.name}: {e}")

    def _infer_single_image(self, img_path, save_dir, config):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None: return

        # 1. 记录原始尺寸 (关键！)
        h_orig, w_orig = image_bgr.shape[:2]
        
        # 2. 预处理缩放 (输入给模型的尺寸)
        process_bgr = image_bgr
        resize_to = config.get('resize')
        
        if resize_to is not None:
            scale = resize_to / max(h_orig, w_orig)
            if scale < 1.0:
                new_w, new_h = int(w_orig * scale), int(h_orig * scale)
                # 强制对齐 Patch Size (14)
                new_w = (new_w // 14) * 14
                new_h = (new_h // 14) * 14
                process_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # 即使不缩放，也要保证是 14 的倍数，否则 ViT 会报错或会有 Padding
            new_w = (w_orig // 14) * 14
            new_h = (h_orig // 14) * 14
            if (new_w, new_h) != (w_orig, h_orig):
                process_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 3. 模型推理
        image_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float().div(255.0).permute(2, 0, 1).to(self.device).unsqueeze(0)
        if self.fp16: image_tensor = image_tensor.half()

        with torch.no_grad():
            if hasattr(self.model, 'infer'):
                output = self.model.infer(image_tensor, resolution_level=9)
            else:
                output = self.model.base_model.model.infer(image_tensor, resolution_level=9)

        # 4. 获取结果 (此时是缩放/裁切后的尺寸)
        depth_pred = output['depth'].squeeze().cpu().numpy() # [h_small, w_small]
        
        # points 通常是 [h, w, 3]，也需要处理
        points_pred = output['points'].squeeze().cpu().numpy() if 'points' in output else None

        # ================= 🔥 关键修改：强制还原回原图尺寸 🔥 =================
        h_pred, w_pred = depth_pred.shape
        if (h_pred, w_pred) != (h_orig, w_orig):
            # 使用双线性插值还原深度图 (平滑)
            depth_pred = cv2.resize(depth_pred, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            
            # 如果有点云，也需要还原
            if points_pred is not None:
                points_pred = cv2.resize(points_pred, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        # ======================================================================

        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存图片 (可选：保存原图还是缩放图？通常保存原图方便对比，或者都不保存省空间)
        # cv2.imwrite(str(save_dir / 'image.jpg'), image_bgr) 
        
        # 保存 NPY (现在是原始尺寸了，可以直接和 GT 对比)
        np.save(str(save_dir / 'depth.npy'), depth_pred)
        
        # 如果需要保存点云
        # if points_pred is not None:
        #     np.save(str(save_dir / 'points.npy'), points_pred)
        
        if 'metric_scale' in output:
            val = output['metric_scale']
            if isinstance(val, torch.Tensor): val = val.float().cpu().item()
            with open(save_dir / 'scale.json', 'w') as f:
                json.dump({'metric_scale': val}, f, indent=4)

        if 'intrinsics' in output:
            import utils3d
            intrinsics = output['intrinsics'].cpu().numpy()[0]
            fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
            with open(save_dir / 'fov.json', 'w') as f:
                json.dump({'fov_x': float(np.rad2deg(fov_x)), 'fov_y': float(np.rad2deg(fov_y))}, f, indent=4)

        if config.get('save_maps', True):
            self._save_visualization(depth_pred, save_dir / 'depth_vis.png')

    def _save_visualization(self, depth, save_path):
        valid_mask = (depth > 0) & np.isfinite(depth)
        if valid_mask.sum() == 0: return
        vmin, vmax = np.percentile(depth[valid_mask], 2), np.percentile(depth[valid_mask], 98)
        plt.figure(figsize=(8, 6))
        plt.imshow(depth, cmap='Spectral_r', vmin=vmin, vmax=vmax) # 推荐 Spectral_r
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=80, bbox_inches='tight')
        plt.close()

# ================= 🚀 2. 主程序 (集成并行逻辑) =================

def main():
    args = setup_args()
    
    # 注入全局变量
    global shard_id
    shard_id = args.shard_id

    # 🔥 重要：在 import moge 之前设置 path
    if args.project_root not in sys.path:
        sys.path.insert(0, args.project_root)
        print(f"🐍 [Env] Added project root: {args.project_root}")
    
    clean_model_path = args.model_path.strip()
    clean_config_path = args.config.strip()
    
    if not os.path.isfile(clean_model_path):
        print(f"❌ [Error] 模型文件不存在: '{clean_model_path}'")
        sys.exit(1)

    # 1. 初始化引擎 (LoRA Mode)
    config = {"resize": args.resize, "overwrite": args.overwrite, "save_maps": True}
    
    # 🔥🔥🔥 实例化 LoRA 引擎
    engine = MoGeInferenceEngine(
        config_path=clean_config_path, 
        checkpoint_path=clean_model_path, 
        device=args.device, 
        fp16=True
    )

    # 2. 扫描任务
    if args.shard_id == 0:
        print(f"📂 Scanning input root: {args.input_root}")
        
    CANDIDATE_IMG_DIRS = ["rgbs", "images_downsampled", "images", "img", "rgb", "image"]
    VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png')
    tasks = []

    if os.path.exists(args.input_root):
        scene_names = sorted(os.listdir(args.input_root))
        for scene_name in scene_names:
            scene_path = os.path.join(args.input_root, scene_name)
            if not os.path.isdir(scene_path): continue
            
            input_rgbs_path = None
            # 优先查找子文件夹
            for candidate in CANDIDATE_IMG_DIRS:
                p = os.path.join(scene_path, candidate)
                if os.path.isdir(p) and len(os.listdir(p)) > 0:
                    input_rgbs_path = p
                    break
            
            # 没找到子文件夹，检查根目录是否有图片
            if not input_rgbs_path:
                 if any(f.endswith(VALID_IMG_EXTS) for f in os.listdir(scene_path)):
                     input_rgbs_path = scene_path

            if input_rgbs_path:
                tasks.append({"input": input_rgbs_path, "output": os.path.join(args.output_root, scene_name)})

    if args.shard_id == 0:
        print(f"✅ 总任务数: {len(tasks)} | 并行分片: {args.total_shards}")

    # 3. 执行任务 (并行分片)
    my_tasks_count = 0
    for i, task in enumerate(tasks):
        # 核心：取模运算分配任务
        if i % args.total_shards != args.shard_id:
            continue
            
        print(f"[Worker {args.shard_id}] 处理: {os.path.basename(task['output'])}")
        engine.process_scene(task['input'], task['output'], config)
        my_tasks_count += 1
        
        if my_tasks_count % 10 == 0: torch.cuda.empty_cache()

    print(f"🎉 [Worker {args.shard_id}] 完成！共处理 {my_tasks_count} 个场景。")

if __name__ == "__main__":
    main()