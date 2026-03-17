import os
import sys
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 防止在无界面的服务器上报错
import matplotlib.pyplot as plt
from typing import *

from peft import LoraConfig, get_peft_model
import matplotlib.patheffects as patheffects
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ================= 1. 动态环境配置 =================
def setup_moge_path():
    current_path = Path(__file__).resolve()
    for parent in [current_path.parents[0], current_path.parents[1], current_path.parents[2]]:
        if (parent / "moge").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    print("⚠️ Warning: 未能自动定位 moge 包路径。")

setup_moge_path()

try:
    from moge.model import import_model_class_by_version
    import utils3d
except ImportError:
    sys.path.insert(0, "/home/szq/moge2/MoGe") 
    from moge.model import import_model_class_by_version
    import utils3d

# ================= 2. 用户配置区域 =================

# 🔥 通用配置：支持单张绝对路径，或多张绝对路径的列表
INPUT_IMAGE_PATHS = [
    "/data1/szq/aaa-vis/scene_001.jpg",
    "/data1/szq/aaa-vis/test_drone_view.png"
] 
# 如果只测单张，可以写成：INPUT_IMAGE_PATHS = "/data1/szq/aaa-vis/single_image.jpg"

OUTPUT_BASE_DIR = "/data1/szq/aaa-vis/out-metric13"            # 结果输出总目录
BATCH_SIZE = 4 # 如果依然爆显存，可以调小为 2 或 1
RESIZE = 0  # 设置为 0 或 None 代表原图尺寸

# 模型特有配置
BASE_TRAIN_CONFIG = "/home/szq/moge2/MoGe/configs/Final_train/config-lora-all.json"

# 终极配置区：将你要对比的模型写在这里！支持 Base 和 LoRA 混合
MODELS_TO_TEST = [
    {
        "name": "Baseline", 
        "type": "base", 
        "path": "/home/szq/moge2/MoGe/vitl-normal.pt",
        "version": "v2"
    },
    {
        "name": "Full Fine-tuning", 
        "type": "base", 
        "path": "/data1/szq/workspace/final-all/checkpoint/00005400_ema.pt",
        "version": "v2"
    },
    {
        "name": "Freeze Vit Backbone", 
        "type": "base", 
        "path": "/data1/szq/workspace/final-neck-lossfine/checkpoint/00003600_ema.pt",
        "version": "v2"
    },
    {
        "name": "Scale Head Only", 
        "type": "base", 
        "path": "/data1/szq/workspace/final-neck-lossfine/checkpoint/00003600_ema.pt",
        "version": "v2"
    },
    {
        "name": "LoRA_Rank64", 
        "type": "lora", 
        "path": "/data1/szq/workspace/lora-batch64-128-with-UElr2/checkpoint/00000600.pt",
        "rank": 64, 
        "alpha": 128
    },
    {
        "name": "LoRA_Rank96", 
        "type": "lora", 
        "path": "/data1/szq/workspace/lora-batch96-192-with-UElr2/checkpoint/00000600.pt",
        "rank": 96, 
        "alpha": 192
    }
]

# ================= 3. 增强版可视化工具函数 (适配尺度深度 Metric Depth) =================

def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth_masked = np.where(depth > 1e-3, depth, np.nan)
    else:
        depth_masked = np.where((depth > 1e-3) & mask, depth, np.nan)
    
    if np.isnan(depth_masked).all():
        return np.zeros((*depth_masked.shape, 3), dtype=np.uint8)

    if normalize:
        min_d, max_d = np.nanquantile(depth_masked, 0.001), np.nanquantile(depth_masked, 0.99)
        if max_d > min_d:
            norm_depth = (depth_masked - min_d) / (max_d - min_d)
        else:
            norm_depth = np.zeros_like(depth_masked)
    else:
        norm_depth = depth_masked
            
    colored = np.nan_to_num(matplotlib.colormaps[cmap](norm_depth)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored

def save_depth_with_colorbar(depth_array: np.ndarray, save_path: str, mask: np.ndarray = None, title: str = None):
    if mask is None:
        mask = (depth_array > 1e-3) & np.isfinite(depth_array)
    else:
        mask = mask & (depth_array > 1e-3) & np.isfinite(depth_array)
        
    depth_masked = np.where(mask, depth_array, np.nan)
    
    if np.isnan(depth_masked).all():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Invalid Depth Map', ha='center', va='center')
        ax.axis('off')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return

    vmin, vmax = np.nanquantile(depth_masked, 0.001), np.nanquantile(depth_masked, 0.99)
    
    h, w = depth_array.shape
    aspect = w / h
    fig_w = max(5, 4 * aspect)
    
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    im = ax.imshow(depth_masked, cmap='Spectral', vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    if title:
        ax.set_title(title)
        
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (m)')
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def colorize_normal(normal: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal

# ================= 4. 核心推理引擎 =================

class MixedInferenceEngine:
    def __init__(self, model_config, device="cuda", fp16=True):
        self.device = torch.device(device)
        self.fp16 = fp16
        self.is_lora = model_config.get("type", "base").lower() == "lora"
        
        print(f"\n📦 [引擎初始化] 模式: {'LoRA' if self.is_lora else 'Base'}, 名称: {model_config['name']}")
        
        if self.is_lora:
            self._init_lora(model_config)
        else:
            self._init_base(model_config)

        self.model.to(self.device)
        self.model.eval()
        if self.fp16: 
            self.model.half()
            print("   ⚡ FP16 模式已开启。")

    def _init_base(self, config):
        model_path = config["path"]
        version = config.get("version", "v2")
        MoGeModel = import_model_class_by_version(version)
        self.model = MoGeModel.from_pretrained(model_path)
        print("   ✅ Base 模型加载完成。")

    def _init_lora(self, config):
        lora_path = config["path"]
        rank = config["rank"]
        alpha = config["alpha"]
        
        with open(BASE_TRAIN_CONFIG, 'r') as f:
            train_config = json.load(f)
        
        model_version = train_config.get('model_version', 'v2')
        MoGeModel = import_model_class_by_version(model_version)
        self.model = MoGeModel(**train_config['model'])
        
        LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]
        HEADS_TO_SAVE = ["scale_head"] 
        peft_config = LoraConfig(
            r=rank, lora_alpha=alpha, bias="none",
            target_modules=LORA_TARGETS, modules_to_save=HEADS_TO_SAVE 
        )
        self.model = get_peft_model(self.model, peft_config)
        
        checkpoint = torch.load(lora_path, map_location='cpu', weights_only=True)
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
        print(f"   ✅ LoRA (Rank {rank}) 权重加载完毕，非LoRA缺失键数: {len(real_missing)}")

    def process_paths(self, image_paths: List[Path], output_path: Path, batch_size=4, resize_to=None):
        """核心推理逻辑：直接接收图片绝对路径列表"""
        if not image_paths: return 0
            
        success_count = 0
        with torch.inference_mode():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Infer & Vis", leave=False):
                batch_paths = image_paths[i:i+batch_size]
                try:
                    count = self._process_batch(batch_paths, output_path, resize_to)
                    success_count += count
                except Exception as e:
                    import traceback
                    print(f"\n❌ Error near {batch_paths[0].name}: {e}")
                    traceback.print_exc()
        return success_count

    def _process_batch(self, batch_paths, root_out, resize_to):
        images_info = []
        max_h, max_w = 0, 0
        
        for img_path in batch_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            h_orig, w_orig = img_bgr.shape[:2]
            
            if resize_to is not None and resize_to > 0:
                scale = resize_to / max(h_orig, w_orig)
                if scale < 1.0:
                    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
                else:
                    new_w, new_h = w_orig, h_orig
            else:
                new_w, new_h = w_orig, h_orig
                
            new_w = max(14, (new_w // 14) * 14)
            new_h = max(14, (new_h // 14) * 14)
            
            if (new_w, new_h) != (w_orig, h_orig):
                process_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                process_bgr = img_bgr
                
            img_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).float().div(255.0).permute(2, 0, 1) 
            
            max_h = max(max_h, new_h)
            max_w = max(max_w, new_w)
            
            images_info.append({
                "path": img_path, "orig_shape": (h_orig, w_orig),
                "new_shape": (new_h, new_w), "tensor": tensor,
                "orig_bgr": img_bgr 
            })
            
        if not images_info: return 0
        actual_batch_size = len(images_info)
        
        batched_tensor = torch.zeros((actual_batch_size, 3, max_h, max_w), device=self.device)
        if self.fp16: batched_tensor = batched_tensor.half()
        
        for i, info in enumerate(images_info):
            h, w = info["new_shape"]
            batched_tensor[i, :, :h, :w] = info["tensor"].to(self.device)
            
        # --- 批量推理 ---
        if hasattr(self.model, 'infer'): 
            output = self.model.infer(batched_tensor)
        elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model.model, 'infer'): 
            output = self.model.base_model.model.infer(batched_tensor)
        else: 
            raise AttributeError("无法找到 infer 方法")
            
        depths = output.get('depth', None)
        normals = output.get('normal', None)
        masks = output.get('mask', None)

        if depths is not None: depths = depths.cpu().numpy()
        if normals is not None: normals = normals.cpu().numpy()
        if masks is not None: masks = masks.cpu().numpy()

        for i, info in enumerate(images_info):
            h_orig, w_orig = info["orig_shape"]
            h_new, w_new = info["new_shape"]
            
            save_dir = root_out / info["path"].stem
            save_dir.mkdir(parents=True, exist_ok=True)
            
            depth_i = None
            if depths is not None:
                d_i = depths[i]
                if d_i.ndim == 3 and d_i.shape[0] == 1: d_i = d_i[0]
                elif d_i.ndim == 3 and d_i.shape[2] == 1: d_i = d_i[:, :, 0]
                depth_i = d_i[:h_new, :w_new].astype(np.float32)
                if depth_i.shape != (h_orig, w_orig):
                    depth_i = cv2.resize(depth_i, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            
            mask_i = None
            if masks is not None:
                m_i = masks[i]
                if m_i.ndim == 3 and m_i.shape[0] == 1: m_i = m_i[0]
                elif m_i.ndim == 3 and m_i.shape[2] == 1: m_i = m_i[:, :, 0]
                mask_i = m_i[:h_new, :w_new].astype(np.float32)
                if mask_i.shape != (h_orig, w_orig):
                    mask_i = cv2.resize(mask_i, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                mask_i = mask_i > 0.5 
            
            normal_i = None
            if normals is not None:
                n_i = normals[i]
                if n_i.shape[0] == 3: 
                    normal_i = n_i[:, :h_new, :w_new].transpose(1, 2, 0).astype(np.float32)
                else: 
                    normal_i = n_i[:h_new, :w_new, :].astype(np.float32)
                if normal_i.shape[:2] != (h_orig, w_orig):
                    normal_i = cv2.resize(normal_i, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            
            cv2.imwrite(str(save_dir / 'image.jpg'), info["orig_bgr"])
            
            if depth_i is not None:
                np.save(str(save_dir / 'depth.npy'), depth_i)
                save_depth_with_colorbar(depth_i, str(save_dir / 'depth_vis.png'), mask_i, title="Predicted Depth (m)")
            
            if normal_i is not None:
                norm_vis = colorize_normal(normal_i, mask_i)
                cv2.imwrite(str(save_dir / 'normal_vis.png'), cv2.cvtColor(norm_vis, cv2.COLOR_RGB2BGR))

        return actual_batch_size

# ================= 5. 下游大图拼接任务 (已适配列表输入) =================

def generate_comparison_grids(image_paths: List[Path], output_base_dir, models_config):
    print(f"\n{'='*60}\n🎨 开始生成学术级对比大图 (高级HUD引线 + 白底黑字UI)\n{'='*60}")
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
    plt.rcParams['font.size'] = 16          
    plt.rcParams['axes.titlesize'] = 18     
    
    output_base_dir = Path(output_base_dir)
    comp_dir = output_base_dir / "Comparisons_Grid"
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(image_paths, desc="Generating Journal Grids"):
        stem = img_path.stem
        parent_dir = img_path.parent # 获取原始图片所在目录
        
        # --- 步骤 A: 加载数据并计算统一色阶 ---
        # 假设 GT npy 文件和原图保存在同一个目录下
        gt_path = parent_dir / f"{stem}.npy"
        gt_depth = np.load(gt_path).squeeze() if gt_path.exists() else None
        
        preds = {}
        all_valid_depths = []
        
        if gt_depth is not None:
            mask = (gt_depth > 1e-3) & np.isfinite(gt_depth)
            if np.any(mask): all_valid_depths.append(gt_depth[mask])
            
        for model in models_config:
            pred_path = output_base_dir / model['name'] / stem / "depth.npy"
            if pred_path.exists():
                p_depth = np.load(pred_path).squeeze()
                preds[model['name']] = p_depth
                mask = (p_depth > 1e-3) & np.isfinite(p_depth)
                if np.any(mask): all_valid_depths.append(p_depth[mask])
                
        if not all_valid_depths:
            continue
            
        global_all_depths = np.concatenate(all_valid_depths)
        global_vmin = np.nanquantile(global_all_depths, 0.01)
        global_vmax = np.nanquantile(global_all_depths, 0.99)
        
        # --- 步骤 B: 寻找探针点 ---
        test_points = []
        if gt_depth is not None:
            mask = (gt_depth > 1e-3) & np.isfinite(gt_depth)
            ys, xs = np.where(mask)
            if len(ys) >= 2:
                idx1, idx2 = len(ys) // 4, 3 * len(ys) // 4  
                test_points = [(xs[idx1], ys[idx1]), (xs[idx2], ys[idx2])]

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
        
        plot_items = [("Input RGB", img_rgb, True)]
        plot_items.append(("Ground Truth", gt_depth, False))
        for model in models_config:
            m_name = model['name']
            short_name = m_name.replace("Original_Base_Model", "Base").replace("LoRA_Rank", "LoRA-")
            plot_items.append((short_name, preds.get(m_name, None), False))

        colors = ['#FF3366', '#00E676']
        
        def draw_probes(ax, data, is_rgb_flag):
            for pt_idx, (x, y) in enumerate(test_points):
                c = colors[pt_idx % len(colors)]
                
                ax.plot(x, y, marker='o', markerfacecolor=c, markeredgecolor='white', 
                        markersize=14, markeredgewidth=2.5, zorder=5)
                
                if is_rgb_flag:
                    text_str = f"P{pt_idx+1}"
                    text_color = c 
                    bg_color = "#555555" 
                    bg_alpha = 0.8
                    arrow_color = 'white'
                else:
                    val = data[y, x]
                    text_str = f"{val:.1f}m"
                    text_color = 'black'    
                    bg_color = 'white'      
                    bg_alpha = 0.85         
                    arrow_color = 'black'   

                x_offset = 65  
                y_offset = -55 if pt_idx == 0 else 55 
                
                ax.annotate(
                    text_str,
                    xy=(x, y), xycoords='data',
                    xytext=(x + x_offset, y + y_offset), textcoords='data',
                    color=text_color, fontsize=15, fontweight='bold',
                    ha='left', va='center', zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_color, edgecolor="none", alpha=bg_alpha),
                    arrowprops=dict(arrowstyle="-", color=arrow_color, linewidth=1.5, alpha=0.8, 
                                    shrinkA=8, shrinkB=2)
                )

        # --- 步骤 C: 导出独立源文件 ---
        element_dir = comp_dir / f"{stem}_Source_Elements"
        element_dir.mkdir(exist_ok=True)
        
        for name, data, is_rgb in plot_items:
            if data is None: continue
            
            h, w = data.shape[:2]
            fig_clean, ax_clean = plt.subplots(figsize=(w/100, h/100), dpi=150)
            fig_clean.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax_clean.axis('off')
            
            if is_rgb:
                ax_clean.imshow(data)
            else:
                mask = (data > 1e-3) & np.isfinite(data)
                data_masked = np.where(mask, data, np.nan)
                ax_clean.imshow(data_masked, cmap='Spectral', vmin=global_vmin, vmax=global_vmax)
            
            draw_probes(ax_clean, data, is_rgb)
                
            safe_name = name.replace(" ", "_").replace("/", "_")
            fig_clean.savefig(element_dir / f"{safe_name}.png", pad_inches=0, transparent=True)
            plt.close(fig_clean)

        # --- 步骤 D: 绘制期刊标准 横向极简排版 ---
        cols = 4
        rows = (len(plot_items) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.8))
        axes = axes.flatten()
        
        for ax_idx, ax in enumerate(axes):
            if ax_idx < len(plot_items):
                name, data, is_rgb = plot_items[ax_idx]
                if data is None:
                    ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center')
                    ax.axis('off')
                    continue
                
                if is_rgb:
                    ax.imshow(data)
                else:
                    mask = (data > 1e-3) & np.isfinite(data)
                    data_masked = np.where(mask, data, np.nan)
                    ax.imshow(data_masked, cmap='Spectral', vmin=global_vmin, vmax=global_vmax)
                
                ax.set_title(name, pad=10, fontsize=18, fontweight='bold')
                ax.axis('off')
                
                draw_probes(ax, data, is_rgb) 
            else:
                ax.axis('off')

        # --- 步骤 E: 添加全局统一标尺 ---
        norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)
        sm = cm.ScalarMappable(cmap='Spectral', norm=norm) 
        sm.set_array([])
        
        fig.subplots_adjust(left=0.01, right=0.91, bottom=0.02, top=0.92, wspace=0.03, hspace=0.10)
        
        cbar_ax = fig.add_axes([0.92, 0.05, 0.012, 0.87]) # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=14, length=0) 
        
        cbar.set_label('Absolute Depth (m)', fontsize=18, fontweight='bold', labelpad=15)
        
        fig.savefig(comp_dir / f"{stem}_Journal_Grid.png", dpi=300, bbox_inches='tight', transparent=False)
        plt.close(fig)

# ================= 6. 主程序启动 =================

def resolve_paths(targets) -> List[Path]:
    """支持解析单一字符串路径或列表，并过滤不存在的文件"""
    resolved = []
    if isinstance(targets, str):
        targets = [targets]
    for t in targets:
        p = Path(t).resolve()
        if p.exists() and p.is_file():
            resolved.append(p)
        else:
            print(f"⚠️ 警告: 图片路径不存在被跳过: {p}")
    return resolved

if __name__ == "__main__":
    print(f"{'='*60}\n🚀 启动混合对比评估可视化流水线\n{'='*60}")
    
    # 解析输入路径
    target_paths = resolve_paths(INPUT_IMAGE_PATHS)
    if not target_paths:
        print("❌ 错误：没有找到有效的输入图片，请检查 INPUT_IMAGE_PATHS 配置。")
        sys.exit(1)
        
    print(f"✅ 找到 {len(target_paths)} 张待推理图片。")
    real_resize = None if RESIZE <= 0 else RESIZE

    # --- 阶段 1：遍历跑完所有模型的推理 ---
    for task_config in MODELS_TO_TEST:
        exp_name = task_config["name"]
        ckpt_path = task_config["path"]
        
        print(f"\n▶️ 开始处理实验: [{exp_name}]")
        if not os.path.exists(ckpt_path):
            print(f"❌ 找不到权重文件，跳过: {ckpt_path}")
            continue
            
        current_out_dir = Path(OUTPUT_BASE_DIR) / exp_name
        current_out_dir.mkdir(parents=True, exist_ok=True)
        
        engine = MixedInferenceEngine(
            model_config=task_config,
            device="cuda"
        )
        
        processed = engine.process_paths(
            image_paths=target_paths,
            output_path=current_out_dir,
            batch_size=BATCH_SIZE,
            resize_to=real_resize
        )
        
        print(f"🎉 实验 [{exp_name}] 完成！共处理了 {processed} 张图片。")
        
        del engine.model
        del engine
        torch.cuda.empty_cache()

    # --- 阶段 2：融合大图下游任务 ---
    generate_comparison_grids(
        image_paths=target_paths,
        output_base_dir=OUTPUT_BASE_DIR,
        models_config=MODELS_TO_TEST
    )
    print(f"\n✅ 所有任务执行完毕！请去 {OUTPUT_BASE_DIR}/Comparisons_Grid 查看合并的对比大图！")