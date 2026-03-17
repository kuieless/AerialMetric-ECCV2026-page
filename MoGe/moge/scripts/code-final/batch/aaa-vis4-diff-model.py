

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

# 通用配置
INPUT_IMAGE_FOLDER = "/data1/szq/Val/Viss/Sup/wild" # 测试图片目录
OUTPUT_BASE_DIR = "/data1/szq/Val/Viss/Sup/mogeout-wild"           # 结果输出总目录
BATCH_SIZE = 1 
RESIZE = 0  

BASE_TRAIN_CONFIG = "/home/szq/moge2/MoGe/configs/Final_train/config-lora-all.json"

MODELS_TO_TEST = [
    {
        "name": "Baseline", 
        "type": "base", # 指定为普通模型
        "path": "/home/szq/moge2/MoGe/vitl-normal.pt",
        "version": "v2"
    },
        {
        "name": "LoRA_Rank96", 
        "type": "lora", # 指定为 LoRA 模型
        "path": "/data1/szq/workspace/lora-batch96-192-with-UElr2/checkpoints/00001000.pt",
        "rank": 96, 
        "alpha": 192
    }

    #    {
    #     "name": "Baseline", 
    #     "type": "base", 
    #     "path": "/home/szq/moge2/MoGe/vitl-normal.pt",
    #     "version": "v2"
    # },
    # {
    #     "name": "Full Fine-tuning", 
    #     "type": "base", 
    #     "path": "/data1/szq/workspace/final-all/checkpoint/00008400_ema.pt",
    #     "version": "v2"
    # },
    # {
    #     "name": "Freeze Vit Backbone", 
    #     "type": "base", 
    #     "path": "/data1/szq/workspace/final-neck-lossfine/checkpoint/00004800_ema.pt",
    #     "version": "v2"
    # },
    # {
    #     "name": "Scale Head Only", 
    #     "type": "base", 
    #     "path": "/data1/szq/workspace/final-head/checkpoint/00010800_ema.pt",
    #     "version": "v2"
    # },
    # {
    #     "name": "LoRA_Rank64", 
    #     "type": "lora", 
    #     "path": "/data1/szq/workspace/lora-batch64-128-with-UElr2/checkpoint/00001300.pt",
    #     "rank": 64, 
    #     "alpha": 128
    # },
    # {
    #     "name": "LoRA_Rank96", 
    #     "type": "lora", 
    #     "path": "/data1/szq/workspace/lora-batch96-192-with-UElr2/checkpoints/00001000.pt",
    #     "rank": 96, 
    #     "alpha": 192
    # }
]

# ================= 3. 增强版可视化工具函数 (适配尺度深度 Metric Depth) =================

def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    """普通的 OpenCV 导出彩色深度图 (不再使用视差倒数)"""
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
            
    # Spectral 映射: 0(近) -> 红色, 1(远) -> 蓝色
    colored = np.nan_to_num(matplotlib.colormaps[cmap](norm_depth)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored

# def save_depth_with_colorbar(depth_array: np.ndarray, save_path: str, mask: np.ndarray = None, title: str = None):
#     """保存带有标尺(Colorbar)的绝对深度图"""
#     if mask is None:
#         mask = (depth_array > 1e-3) & np.isfinite(depth_array)
#     else:
#         mask = mask & (depth_array > 1e-3) & np.isfinite(depth_array)
        
#     # 直接使用深度，不再取倒数
#     depth_masked = np.where(mask, depth_array, np.nan)
    
#     if np.isnan(depth_masked).all():
#         fig, ax = plt.subplots()
#         ax.text(0.5, 0.5, 'Invalid Depth Map', ha='center', va='center')
#         ax.axis('off')
#         fig.savefig(save_path, bbox_inches='tight')
#         plt.close(fig)
#         return

#     vmin, vmax = np.nanquantile(depth_masked, 0.001), np.nanquantile(depth_masked, 0.99)
    
#     h, w = depth_array.shape
#     aspect = w / h
#     fig_w = max(5, 4 * aspect)
    
#     fig, ax = plt.subplots(figsize=(fig_w, 4))
#     # cmap='Spectral': 深度小(近)呈现红色，深度大(远)呈现蓝色
#     im = ax.imshow(depth_masked, cmap='Spectral', vmin=vmin, vmax=vmax)
#     ax.axis('off')
    
#     if title:
#         ax.set_title(title)
        
#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label('Depth (m)')  # 明确标注为绝对深度
    
#     plt.tight_layout()
#     fig.savefig(save_path, bbox_inches='tight', dpi=150)
#     plt.close(fig)
def save_depth_clean(depth_array: np.ndarray, save_path: str, mask: np.ndarray = None, target_size=(1280, 960)):
    """
    保存固定尺寸、无标尺、无白边的纯净彩色深度图
    """
    if mask is None:
        mask = (depth_array > 1e-3) & np.isfinite(depth_array)
    
    # 1. 强制缩放到统一尺寸 (确保所有模型的图在文件夹里看起来一样大)
    depth_resized = cv2.resize(depth_array, target_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST) > 0
    
    depth_masked = np.where(mask_resized, depth_resized, np.nan)
    
    # 2. 计算量程 (使用 1-99% 避免噪点干扰颜色)
    if not np.all(np.isnan(depth_masked)):
        vmin = np.nanquantile(depth_masked, 0.01)
        vmax = np.nanquantile(depth_masked, 0.99)
    else:
        vmin, vmax = 0, 10

    # 3. 绘图：关键在于去除所有装饰和白边
    # 设置 figsize 确保像素级对齐
    fig, ax = plt.subplots(figsize=(target_size[0]/100, target_size[1]/100), dpi=100)
    
    # 强制填满，不留任何空白
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    ax.imshow(depth_masked, cmap='Spectral', vmin=vmin, vmax=vmax)
    ax.axis('off') # 关掉坐标轴
    
    # savefig 时 pad_inches=0 极其重要，否则会有白边
    fig.savefig(save_path, pad_inches=0)
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

    def process_folder(self, input_path, output_path, batch_size=4, resize_to=None):
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # --- 修改处：只读 jpg/jpeg 作为推理输入，避开 png ---
        # 或者你可以使用 [p for p in input_path.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'}]
        image_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'}])
        
        if not image_paths: 
            print(f"⚠️ 在 {input_path} 中没找到 JPG 图片，请检查路径。")
            return 0
            
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
            
            # 保存原图
            cv2.imwrite(str(save_dir / 'image.jpg'), info["orig_bgr"])
            
            # 🔥 新的带标尺的深度图保存方法 (尺度深度)
            if depth_i is not None:
                np.save(str(save_dir / 'depth.npy'), depth_i)
                # save_depth_with_colorbar(depth_i, str(save_dir / 'depth_vis.png'), mask_i, title="Predicted Depth (m)")
                save_depth_clean(depth_i, str(save_dir / 'depth_vis.png'), mask_i)
            # 传统法向图
            if normal_i is not None:
                norm_vis = colorize_normal(normal_i, mask_i)
                cv2.imwrite(str(save_dir / 'normal_vis.png'), cv2.cvtColor(norm_vis, cv2.COLOR_RGB2BGR))

        return actual_batch_size

# ================= 5. 下游大图拼接任务 =================

# ================= 5. 下游大图拼接任务 (学术期刊双栏排版 - 最终优化版) =================
import matplotlib.patheffects as patheffects
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ================= 5. 下游大图拼接任务 (学术期刊极简高级排版 - 悬浮引线版) =================
def render_item_standardized(ax, name, data, is_rgb, ref_range, test_points, gt_data=None):
    """
    统一渲染函数：确保大图和单图的色彩、探针、样式完全一致
    ref_range: (vmin, vmax) 强制使用的色彩映射范围
    """
    vmin, vmax = ref_range
    h, w = data.shape[:2]
    
    if is_rgb:
        ax.imshow(data)
    else:
        mask = (data > 1e-3) & np.isfinite(data)
        # 稀疏点云 GT 渲染
        if "Ground Truth" in name or data.size < (h * w * 0.1): # 如果是稀疏数据
            ys, xs = np.where(mask)
            ax.set_facecolor('white') 
            ax.scatter(xs, ys, c=data[mask], cmap='Spectral', 
                       vmin=vmin, vmax=vmax, s=15, edgecolors='none')
            ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect('equal')
        else:
            # 稠密深度图：强制使用参考量程
            # 这样预测偏小会变红，偏大会变蓝
            data_masked = np.where(mask, data, np.nan)
            ax.imshow(data_masked, cmap='Spectral', vmin=vmin, vmax=vmax)

    # 统一调用带对比功能的探针
    draw_probes_with_comparison(ax, data, is_rgb, test_points, gt_data)
    ax.axis('off')
def render_unified_item(ax, name, data, is_rgb, ref_range, test_points, global_params):
    """
    统一渲染入口：确保单图(Elements)和大图(Grid)的视觉表现像素级对齐
    """
    vmin, vmax = ref_range
    if data is None:
        ax.axis('off')
        return

    h, w = data.shape[:2]
    
    if is_rgb:
        ax.imshow(data)
    else:
        mask = (data > 1e-3) & np.isfinite(data)
        if name == "Ground Truth":
            # --- 统一的散点 GT 渲染 ---
            ys, xs = np.where(mask)
            ax.set_facecolor('white') 
            # s=15 在大图可能略显大，在单图刚好。可以根据需要微调
            ax.scatter(xs, ys, c=data[mask], cmap='Spectral', 
                       vmin=vmin, vmax=vmax, s=15, edgecolors='none', alpha=1.0)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_aspect('equal')
        else:
            # --- 统一的稠密深度渲染 ---
            data_masked = np.where(mask, data, np.nan)
            ax.imshow(data_masked, cmap='Spectral', vmin=vmin, vmax=vmax)

    # 统一绘制探针
    draw_probes(ax, data, is_rgb, test_points)
    ax.axis('off')
def draw_unified_probes(ax, data, is_rgb, test_points):
    """
    统一的探针绘制逻辑：深度图上强制白底黑字，RGB上彩色文字
    """
    colors = ['#FF3366', '#00E676'] # 亮粉和亮绿
    for pt_idx, (x, y) in enumerate(test_points):
        c = colors[pt_idx % len(colors)]
        
        # 1. 绘制靶心
        ax.plot(x, y, marker='o', markerfacecolor=c, markeredgecolor='white', 
                markersize=12, markeredgewidth=2, zorder=10)
        
        # 2. 准备文字
        if is_rgb:
            text_str = f"P{pt_idx+1}"
            text_color = c
            bg_color = "white"
        else:
            val = data[y, x]
            text_str = f"{val:.1f}m"
            text_color = 'black'
            bg_color = 'white'

        # 3. 悬浮标注
        x_offset = 60
        y_offset = -50 if pt_idx == 0 else 50
        ax.annotate(
            text_str, xy=(x, y), xytext=(x + x_offset, y + y_offset),
            color=text_color, fontsize=14, fontweight='bold',
            va='center', zorder=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, edgecolor="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color='white' if is_rgb else 'black', linewidth=1, alpha=0.5)
        )
def render_standard_frame(ax, name, data, is_rgb, ref_range, test_points):
    """
    核心渲染函数：处理所有可视化逻辑
    ref_range: (vmin, vmax)
    """
    vmin, vmax = ref_range
    h, w = data.shape[:2]
    
    if is_rgb:
        ax.imshow(data)
    else:
        mask = (data > 1e-3) & np.isfinite(data)
        # 识别 Ground Truth (通常是稀疏的)
        if "Ground Truth" in name:
            ys, xs = np.where(mask)
            ax.set_facecolor('white') 
            ax.scatter(xs, ys, c=data[mask], cmap='Spectral', 
                       vmin=vmin, vmax=vmax, s=10, edgecolors='none', zorder=2)
            ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect('equal')
        else:
            # 稠密预测图
            data_masked = np.where(mask, data, np.nan)
            ax.imshow(data_masked, cmap='Spectral', vmin=vmin, vmax=vmax, zorder=1)

    # 叠加探针
    if test_points:
        draw_unified_probes(ax, data, is_rgb, test_points)
    
    ax.axis('off')
# def generate_comparison_grids(input_dir, output_base_dir, models_config):
#     print(f"\n{'='*60}\n🎨 开始生成学术级对比大图 (支持 PNG/NPY GT + 毫米转换)\n{'='*60}")
    
#     # --- 1. 全局配置 ---
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
#     plt.rcParams['font.size'] = 16 
    
#     input_dir = Path(input_dir)
#     output_base_dir = Path(output_base_dir)
#     comp_dir = output_base_dir / "Comparisons_Grid"
#     comp_dir.mkdir(parents=True, exist_ok=True)
    
#     valid_exts = {'.jpg', '.png', '.jpeg', '.JPG', '.PNG'}
#     # 过滤出 RGB 图像文件（排除掉可能是 GT 的 png）
#     image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'}])
    
#     for img_path in tqdm(image_paths, desc="Generating Journal Grids"):
#         stem = img_path.stem
        
#         # --- 修改处 B：专门去读 png 作为 Ground Truth 并转换单位 ---
#         gt_depth = None
#         png_gt_path = input_dir / f"{stem}.png"
        
#         if png_gt_path.exists():
#             # 使用 cv2 读取原始数据 (16bit)
#             gt_png = cv2.imread(str(png_gt_path), cv2.IMREAD_UNCHANGED)
#             if gt_png is not None:
#                 # 转换毫米到米
#                 gt_depth = gt_png.astype(np.float32) / 1000.0
#                 # 屏蔽 0 值
#                 gt_depth[gt_depth <= 0] = np.nan
#         else:
#             # 如果没有 png，再尝试找 npy
#             npy_gt_path = input_dir / f"{stem}.npy"
#             if npy_gt_path.exists():
#                 gt_depth = np.load(npy_gt_path).squeeze()
        
#         preds = {}
#         all_valid_depths = []
        
#         # 收集所有有效深度值用于计算全局色彩量程
#         if gt_depth is not None:
#             mask = (gt_depth > 1e-3) & np.isfinite(gt_depth)
#             if np.any(mask): all_valid_depths.append(gt_depth[mask])
            
#         for model in models_config:
#             pred_path = output_base_dir / model['name'] / stem / "depth.npy"
#             if pred_path.exists():
#                 p_depth = np.load(pred_path).squeeze()
#                 preds[model['name']] = p_depth
#                 mask = (p_depth > 1e-3) & np.isfinite(p_depth)
#                 if np.any(mask): all_valid_depths.append(p_depth[mask])
                
#         if not all_valid_depths:
#             continue
            
#         global_all_depths = np.concatenate(all_valid_depths)
#         global_vmin = np.nanquantile(global_all_depths, 0.01)
#         global_vmax = np.nanquantile(global_all_depths, 0.99)
        

#         test_points = []
#         if gt_depth is not None:
#             mask = (gt_depth > 1e-3) & np.isfinite(gt_depth)
#             h, w = mask.shape
            
#             # 定义中心感兴趣区域 (ROI)，例如图像正中间 50% 的区域
#             roi_y_start, roi_y_end = h // 4, 3 * h // 4
#             roi_x_start, roi_x_end = w // 4, 3 * w // 4
            
#             # 创建一个只包含中心区域的临时 Mask
#             center_mask = np.zeros_like(mask)
#             center_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            
#             ys, xs = np.where(center_mask)
            
#             if len(ys) >= 2:
#                 # 在中心区域的有效点中，取两个分布较开的点（比如索引的 1/3 和 2/3 处）
#                 idx1, idx2 = len(ys) // 3, 2 * len(ys) // 3  
#                 test_points = [(xs[idx1], ys[idx1]), (xs[idx2], ys[idx2])]
#             elif len(np.where(mask)[0]) >= 2:
#                 # 如果中心实在没点，再回退到全局找点
#                 ys_all, xs_all = np.where(mask)
#                 idx1, idx2 = len(ys_all) // 4, 3 * len(ys_all) // 4
#                 test_points = [(xs_all[idx1], ys_all[idx1]), (xs_all[idx2], ys_all[idx2])]

#         img_bgr = cv2.imread(str(img_path))
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
        
#         plot_items = [("Input RGB", img_rgb, True)]
#         plot_items.append(("Ground Truth", gt_depth, False))
#         for model in models_config:
#             m_name = model['name']
#             short_name = m_name.replace("Original_Base_Model", "Base").replace("LoRA_Rank", "LoRA-")
#             plot_items.append((short_name, preds.get(m_name, None), False))

#         # --- 统一的探针绘制逻辑 (内部函数，用于复用) ---
#         colors = ['#FF3366', '#00E676'] # 亮粉和亮绿，用于靶心
        
#         def draw_probes(ax, data, is_rgb_flag):
#             for pt_idx, (x, y) in enumerate(test_points):
#                 c = colors[pt_idx % len(colors)]
                
#                 # 绘制靶心：彩色内，白色外描边
#                 ax.plot(x, y, marker='o', markerfacecolor=c, markeredgecolor='white', 
#                         markersize=14, markeredgewidth=2.5, zorder=5)
                
#                 if is_rgb_flag:
#                     text_str = f"P{pt_idx+1}"
#                     text_color = c # RGB图上文字颜色保持和点一致
#                     bg_color = "#555555" # RGB图背景复杂，保持深色底白字可能更好，或者也统一成白底黑字
#                     bg_alpha = 0.8
#                     arrow_color = 'white'
#                 else:
#                     val = data[y, x]
#                     text_str = f"{val:.1f}m"
#                     # 🔥 核心修改区：深度图上强制白底黑字 🔥
#                     text_color = 'black'    # 文字纯黑
#                     bg_color = 'white'      # 背景纯白
#                     bg_alpha = 0.85         # 半透明
#                     arrow_color = 'black'   # 引线黑色

#                 # 距离偏移量
#                 x_offset = 65  
#                 y_offset = -55 if pt_idx == 0 else 55 
                
#                 ax.annotate(
#                     text_str,
#                     xy=(x, y), xycoords='data',
#                     xytext=(x + x_offset, y + y_offset), textcoords='data',
#                     color=text_color, fontsize=15, fontweight='bold',
#                     ha='left', va='center', zorder=6,
#                     # 设置背景框样式
#                     bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_color, edgecolor="none", alpha=bg_alpha),
#                     # 设置引线样式
#                     # arrowprops=dict(arrowstyle="-", color=arrow_color, linewidth=1.5, alpha=0.8, 
#                     #                 shrinkA=8, shrinkB=2)
#                 )

#         # --- 步骤 C: 导出独立源文件 ---
#         element_dir = comp_dir / f"{stem}_Source_Elements"
#         element_dir.mkdir(exist_ok=True)
        
#         for name, data, is_rgb in plot_items:
#             if data is None: continue
            
#             h, w = data.shape[:2]
#             fig_clean, ax_clean = plt.subplots(figsize=(w/100, h/100), dpi=150)
#             fig_clean.subplots_adjust(left=0, right=1, top=1, bottom=0)
#             ax_clean.axis('off')
            
#             # if is_rgb:
#             #     ax_clean.imshow(data)
#             # else:
#             #     mask = (data > 1e-3) & np.isfinite(data)
#             #     data_masked = np.where(mask, data, np.nan)
#             #     ax_clean.imshow(data_masked, cmap='Spectral', vmin=global_vmin, vmax=global_vmax)
            
#             # draw_probes(ax_clean, data, is_rgb)
                
#             # safe_name = name.replace(" ", "_").replace("/", "_")
#             # fig_clean.savefig(element_dir / f"{safe_name}.png", pad_inches=0, transparent=True)
#             # plt.close(fig_clean)
#             if is_rgb:
#                 ax_clean.imshow(data)
#             else:
#                 mask = (data > 1e-3) & np.isfinite(data)
                
#                 # 🔥 针对稀疏 GT 点云的强化渲染 (散点法)
#                 if name == "Ground Truth":
#                     ys, xs = np.where(mask)
#                     vals = data[mask]
                    
#                     ax_clean.set_facecolor('white') # 保持白底
#                     # 使用 scatter，s 控制点的大小(加粗)，这里设为15，你可以根据分辨率微调
#                     ax_clean.scatter(xs, ys, c=vals, cmap='Spectral', vmin=global_vmin, vmax=global_vmax, 
#                                      s=15, alpha=1.0, edgecolors='none')
                    
#                     # 强制对齐图像尺寸与长宽比 (极其重要，否则散点图会变形)
#                     h, w = data.shape
#                     ax_clean.set_xlim(0, w)
#                     ax_clean.set_ylim(h, 0) # 图像系y轴向下，需翻转
#                     ax_clean.set_aspect('equal')
                
#                 # 针对稠密深度图，保留 imshow
#                 else:
#                     data_masked = np.where(mask, data, np.nan)
#                     ax_clean.imshow(data_masked, cmap='Spectral', vmin=global_vmin, vmax=global_vmax)

#         # --- 步骤 D: 绘制期刊标准 横向极简排版 (2行 x 4列) ---

#         cols = 4
#         rows = (len(plot_items) + cols - 1) // cols
        
#         # 设定一个标准显示高度和宽度，确保排版时格子大小完全一样
#         STD_W, STD_H = 5, 4 
#         fig, axes = plt.subplots(rows, cols, figsize=(cols * STD_W, rows * STD_H))
#         axes = axes.flatten()
        
#         for ax_idx, ax in enumerate(axes):
#             if ax_idx < len(plot_items):
#                 name, data, is_rgb = plot_items[ax_idx]
#                 if data is None:
#                     ax.axis('off')
#                     continue
                
#                 # --- 1. 确定基准量程 (以 GT 为准) ---
#                 # 我们需要先获取 GT 的量程作为参考系
#                 if gt_depth is not None:
#                     gt_mask = (gt_depth > 1e-3) & np.isfinite(gt_depth)
#                     valid_gt_data = gt_depth[gt_mask]
#                     # 使用 GT 的 1-99% 作为该场景的标准色阶
#                     ref_vmin = np.nanquantile(valid_gt_data, 0.01)
#                     ref_vmax = np.nanquantile(valid_gt_data, 0.99)
#                 else:
#                     ref_vmin, ref_vmax = global_vmin, global_vmax

#                 # --- 2. 渲染逻辑 ---
#                 if is_rgb:
#                     ax.imshow(data)
#                 else:
#                     mask = (data > 1e-3) & np.isfinite(data)
                    
#                     if name == "Ground Truth":
#                         ys, xs = np.where(mask)
#                         ax.set_facecolor('white') 
#                         # GT 始终作为标准
#                         ax.scatter(xs, ys, c=data[mask], cmap='Spectral', 
#                                 vmin=ref_vmin, vmax=ref_vmax, s=8, edgecolors='none')
                        
#                         h, w = data.shape
#                         ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect('equal')
                    
#                     else:
#                         # 🔥 关键修改：模型预测图强制使用 GT 的 vmin/vmax
#                         # 这样如果模型预测偏小，颜色会整体变红；预测偏大，颜色会整体变蓝
#                         data_masked = np.where(mask, data, np.nan)
#                         ax.imshow(data_masked, cmap='Spectral', vmin=ref_vmin, vmax=ref_vmax)

#                 ax.set_title(name, pad=8, fontsize=16, fontweight='bold')
#                 ax.axis('off')
                
#                 # 绘制探针深度值
#                 draw_probes(ax, data, is_rgb) 
#             else:
#                 ax.axis('off')

#         # --- 步骤 E: 移除原本的全局标尺代码 ---
#         # (删掉之前的 fig.add_axes 和 fig.colorbar 部分)
        
#         fig.tight_layout() # 自动调整间距
#         fig.savefig(comp_dir / f"{stem}_Journal_Grid.png", dpi=300, bbox_inches='tight')
#         plt.close(fig)

#         # --- 步骤 E: 添加全局统一标尺 ---
#         norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)
#         sm = cm.ScalarMappable(cmap='Spectral', norm=norm) 
#         sm.set_array([])
        
#         fig.subplots_adjust(left=0.01, right=0.91, bottom=0.02, top=0.92, wspace=0.03, hspace=0.10)
        
#         cbar_ax = fig.add_axes([0.92, 0.05, 0.012, 0.87]) # [left, bottom, width, height]
#         cbar = fig.colorbar(sm, cax=cbar_ax)
        
#         cbar.outline.set_visible(False)
#         cbar.ax.tick_params(labelsize=14, length=0) 
        
#         cbar.set_label('Absolute Depth (m)', fontsize=18, fontweight='bold', labelpad=15)
        
#         fig.savefig(comp_dir / f"{stem}_Journal_Grid.png", dpi=300, bbox_inches='tight', transparent=False)
#         plt.close(fig)
def generate_comparison_grids(input_dir, output_base_dir, models_config):
    input_dir = Path(input_dir)
    output_base_dir = Path(output_base_dir)
    comp_dir = output_base_dir / "Comparisons_Grid"
    comp_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'}])
    
    for img_path in tqdm(image_paths, desc="Final Visualizing"):
        stem = img_path.stem
        
        # 1. 加载数据 (此处逻辑保持你之前的毫米转米)
        gt_depth = None
        png_gt_path = input_dir / f"{stem}.png"
        if png_gt_path.exists():
            gt_png = cv2.imread(str(png_gt_path), cv2.IMREAD_UNCHANGED)
            if gt_png is not None: gt_depth = gt_png.astype(np.float32) / 1000.0
        
        preds = {}
        all_depths_for_range = []
        if gt_depth is not None: all_depths_for_range.append(gt_depth[gt_depth > 0])
        
        for model in models_config:
            p_path = output_base_dir / model['name'] / stem / "depth.npy"
            if p_path.exists():
                d = np.load(p_path).squeeze()
                preds[model['name']] = d
                all_depths_for_range.append(d[d > 0])
        
        if not all_depths_for_range: continue
        
        # 2. 确定全局色阶 (基于该场景所有模型的并集)
        concat_depths = np.concatenate(all_depths_for_range)
        ref_vmin = np.nanquantile(concat_depths, 0.01)
        ref_vmax = np.nanquantile(concat_depths, 0.99)
        ref_range = (ref_vmin, ref_vmax)

        # 3. 选取探针点
        test_points = []
        if gt_depth is not None:
            mask = (gt_depth > 1e-3) & np.isfinite(gt_depth)
            h, w = mask.shape
            roi = mask[h//4:3*h//4, w//4:3*w//4]
            ys, xs = np.where(mask) # 简化：直接取全局
            if len(ys) > 10:
                idx1, idx2 = len(ys)//3, 2*len(ys)//3
                test_points = [(xs[idx1], ys[idx1]), (xs[idx2], ys[idx2])]

        # 4. 准备待绘制列表
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        plot_items = [("Input RGB", img_rgb, True), ("Ground Truth", gt_depth, False)]
        for model in models_config:
            m_name = model['name']
            short_name = m_name.replace("LoRA_Rank", "LoRA-").replace("Full Fine-tuning", "Full-FT")
            plot_items.append((short_name, preds.get(m_name), False))

        # --- 5. 核心：同时生成大图和独立单图 ---
        element_dir = comp_dir / f"{stem}_Source_Elements"
        element_dir.mkdir(exist_ok=True)
        
        cols = 4
        rows = (len(plot_items) + cols - 1) // cols
        fig_grid, axes_grid = plt.subplots(rows, cols, figsize=(cols*5, rows*4), dpi=200)
        axes_grid = axes_grid.flatten()

        for idx, (name, data, is_rgb) in enumerate(plot_items):
            if data is None: continue
            
            # A. 渲染到大图的子图中
            render_standard_frame(axes_grid[idx], name, data, is_rgb, ref_range, test_points)
            axes_grid[idx].set_title(name, fontsize=16, fontweight='bold', pad=10)
            
            # B. 渲染到独立的高清单图中
            h_px, w_px = data.shape[:2]
            fig_solo, ax_solo = plt.subplots(figsize=(w_px/100, h_px/100), dpi=100)
            fig_solo.subplots_adjust(0,0,1,1) # 彻底去除白边
            
            render_standard_frame(ax_solo, name, data, is_rgb, ref_range, test_points)
            
            safe_name = name.replace(" ", "_")
            fig_solo.savefig(element_dir / f"{safe_name}.png", pad_inches=0)
            plt.close(fig_solo)

        # 6. 保存大图
        for ax in axes_grid[len(plot_items):]: ax.axis('off')
        fig_grid.tight_layout()
        fig_grid.savefig(comp_dir / f"{stem}_Journal_Grid.png", bbox_inches='tight')
        plt.close(fig_grid)
# ================= 6. 主程序启动 =================

if __name__ == "__main__":
    print(f"{'='*60}\n🚀 启动混合对比评估可视化流水线\n{'='*60}")
    print(f"输入文件夹: {INPUT_IMAGE_FOLDER}")
    
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
        
        processed = engine.process_folder(
            input_path=INPUT_IMAGE_FOLDER,
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
        input_dir=INPUT_IMAGE_FOLDER,
        output_base_dir=OUTPUT_BASE_DIR,
        models_config=MODELS_TO_TEST
    )
    print(f"\n✅ 所有任务执行完毕！请去 {OUTPUT_BASE_DIR}/Comparisons_Grid 查看合并的对比大图！")