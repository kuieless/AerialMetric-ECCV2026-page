import os
import sys
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

# ================= 动态环境配置 =================
def setup_moge_path():
    current_path = Path(__file__).resolve()
    for parent in [current_path.parents[0], current_path.parents[1], current_path.parents[2]]:
        if (parent / "moge").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    print("⚠️ Warning: 未能自动定位 moge 包路径，请确保环境变量已设置。")

setup_moge_path()

try:
    from moge.model import import_model_class_by_version
    import utils3d
except ImportError:
    sys.path.insert(0, "/home/szq/moge2/MoGe") 
    from moge.model import import_model_class_by_version
    import utils3d

# ================= 核心类 =================

class MogeLoRAEngine:
    def __init__(self, config_path, lora_path, device="cuda", fp16=True):
        self.device = torch.device(device)
        self.fp16 = fp16
        
        print(f"\n📦 [LoRA Engine] 初始化...")
        print(f"   Config: {config_path}")
        print(f"   LoRA:   {lora_path}")

        with open(config_path, 'r') as f:
            train_config = json.load(f)
        
        model_version = train_config.get('model_version', 'v2')
        print(f"   Base Model Version: {model_version}")
        MoGeModel = import_model_class_by_version(model_version)
        self.model = MoGeModel(**train_config['model'])
        
        # LoRA Config
        LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]
        HEADS_TO_SAVE = ["scale_head"] 
        peft_config = LoraConfig(
            r=96, lora_alpha=192, bias="none",
            target_modules=LORA_TARGETS, modules_to_save=HEADS_TO_SAVE 
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Load Weights
        print("   Loading LoRA weights...")
        checkpoint = torch.load(lora_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = {}
        model_keys = set(self.model.state_dict().keys())
        
        for k, v in state_dict.items():
            if k in model_keys:
                new_state_dict[k] = v; continue
            
            prefixed_k = f"base_model.model.{k}"
            if prefixed_k in model_keys:
                new_state_dict[prefixed_k] = v; continue
            
            parts = prefixed_k.split('.')
            if parts[-1] in ['weight', 'bias']:
                base_injected_k = ".".join(parts[:-1] + ["base_layer", parts[-1]])
                if base_injected_k in model_keys:
                    new_state_dict[base_injected_k] = v; continue
            
            for head in HEADS_TO_SAVE:
                if k.startswith(head):
                    suffix = k[len(head)+1:]
                    trainable_k = f"base_model.model.{head}.modules_to_save.default.{suffix}"
                    if trainable_k in model_keys:
                        new_state_dict[trainable_k] = v; break

        msg = self.model.load_state_dict(new_state_dict, strict=False)
        real_missing = [k for k in msg.missing_keys if "lora_" not in k]
        print(f"✅ LoRA 加载完成。非 LoRA 缺失键: {len(real_missing)}")

        self.model.to(self.device)
        self.model.eval()
        if self.fp16: self.model.half()

    def run_scene(self, input_path, output_path, stride=1, resize=1024, batch_size=4):
        input_path = Path(input_path)
        output_path = Path(output_path)
        valid_exts = {'.jpg', '.png', '.jpeg', '.JPG', '.PNG'}
        image_paths = sorted([p for p in input_path.iterdir() if p.suffix in valid_exts])
        
        if not image_paths: return 0
        if stride > 1: image_paths = image_paths[::stride]
        
        success_count = 0
        with torch.inference_mode():
            # 🔥 将单张循环改为 Batch 分块循环
            for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Scene: {input_path.name}", leave=False):
                batch_paths = image_paths[i:i+batch_size]
                try:
                    count = self._process_batch(batch_paths, output_path, resize)
                    success_count += count
                except Exception as e:
                    print(f"❌ Error processing batch near {batch_paths[0].name}: {e}")
        return success_count

    def _process_batch(self, batch_paths, root_out, resize_to):
        images_info = []
        max_h, max_w = 0, 0
        
        # 1. 读取并处理 Batch 内所有图片的尺寸
        for img_path in batch_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            h_orig, w_orig = img_bgr.shape[:2]
            
            if resize_to is not None:
                scale = resize_to / max(h_orig, w_orig)
                new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            else:
                new_w, new_h = w_orig, h_orig
                
            # 必须对齐 14 (ViT要求)
            new_w = max(14, (new_w // 14) * 14)
            new_h = max(14, (new_h // 14) * 14)
            
            if (new_w, new_h) != (w_orig, h_orig):
                process_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                process_bgr = img_bgr
                
            img_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).float().div(255.0).permute(2, 0, 1) # [3, H, W]
            
            max_h = max(max_h, new_h)
            max_w = max(max_w, new_w)
            
            images_info.append({
                "path": img_path, "orig_shape": (h_orig, w_orig),
                "new_shape": (new_h, new_w), "tensor": tensor
            })
            
        if not images_info: return 0
        
        actual_batch_size = len(images_info)
        
        # 2. 补齐 (Padding) 到 Batch 里的最大尺寸
        batched_tensor = torch.zeros((actual_batch_size, 3, max_h, max_w), device=self.device)
        if self.fp16: batched_tensor = batched_tensor.half()
        
        for i, info in enumerate(images_info):
            h, w = info["new_shape"]
            batched_tensor[i, :, :h, :w] = info["tensor"].to(self.device)
            
        # 3. 批量推理
        if hasattr(self.model, 'infer'): output = self.model.infer(batched_tensor)
        elif hasattr(self.model.base_model.model, 'infer'): output = self.model.base_model.model.infer(batched_tensor)
        else: raise AttributeError("无法找到 infer 方法")
            
        depths = output['depth'].cpu().numpy()
        if depths.ndim == 4: depths = depths.squeeze(1) # [B, 1, H, W] -> [B, H, W]
        
        intrinsics_batch = output.get('intrinsics', None)
        if intrinsics_batch is not None: intrinsics_batch = intrinsics_batch.cpu().numpy()

        # 4. 后处理：精准裁剪并缩放回原图
        for i, info in enumerate(images_info):
            h_orig, w_orig = info["orig_shape"]
            h_new, w_new = info["new_shape"]
            
            # 裁掉之前 Padding 补黑边的部分
            depth = depths[i, :h_new, :w_new]
            
            if depth.shape != (h_orig, w_orig):
                depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                
            # 保存
            save_dir = root_out / info["path"].stem
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(save_dir / 'depth.npy'), depth)
            
            if intrinsics_batch is not None:
                intrinsics = intrinsics_batch[i]
                fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
                with open(save_dir / 'fov.json', 'w') as f:
                    json.dump({'fov_x': float(np.rad2deg(fov_x)), 'fov_y': float(np.rad2deg(fov_y))}, f)
                    
        return actual_batch_size

class DatasetAutoParser:
    def __init__(self):
        self.type_keywords = {'image': ['image', 'images', 'rgb', 'rgbs', 'color', 'jpg']}
        self.scenes = []

    def scan(self, root_path):
        root_path = Path(root_path)
        if not root_path.exists(): return
        print(f"🕵️  Scanning: {root_path} ...")
        for root, dirs, files in os.walk(root_path):
            if not dirs: continue
            img_dir = None
            for d in dirs:
                if any(k in d.lower() for k in self.type_keywords['image']):
                    img_dir = Path(root) / d; break
            if img_dir:
                self.scenes.append({
                    'source_root': root_path,
                    'rel_path': Path(root).relative_to(root_path),
                    'img_dir': img_dir
                })

# ================= 封装的调用接口 =================

def run_inference_pipeline(
    input_roots, output_root, lora_config, lora_weight, 
    sampling_ratio=1.0, resize=1024, device="cuda", batch_size=4 # 🔥 新增 batch_size 参数
):
    parser = DatasetAutoParser()
    if isinstance(input_roots, str): input_roots = [input_roots]
    for root in input_roots: parser.scan(root)
    
    tasks = parser.scenes
    print(f"✅ 共发现 {len(tasks)} 个场景待处理。")
    if not tasks: return

    if not os.path.exists(lora_config) or not os.path.exists(lora_weight):
        print(f"❌ 错误: 找不到 LoRA 文件:\nConfig: {lora_config}\nWeight: {lora_weight}")
        return

    engine = MogeLoRAEngine(lora_config, lora_weight, device=device)
    stride = int(1 / sampling_ratio) if sampling_ratio < 1.0 else 1
    
    for i, task in enumerate(tasks):
        source_name = task['source_root'].name
        output_dir = Path(output_root) / source_name / task['rel_path']
        print(f"▶️ [{i+1}/{len(tasks)}] {task['rel_path']}")
        
        count = engine.run_scene(
            input_path=task['img_dir'],
            output_path=output_dir,
            stride=stride,
            resize=resize,
            batch_size=batch_size # 🔥 传入 batch_size
        )
        print(f"   ✅ 完成 {count} 张")
