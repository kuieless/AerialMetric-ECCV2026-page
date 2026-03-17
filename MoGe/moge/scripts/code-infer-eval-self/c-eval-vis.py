import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ================= 1. 配置区域 =================
BASE_PRED_DIR = "/data1/szq/self1/extracted"

TASKS = [
    {"name": "Campus", "path": os.path.join(BASE_PRED_DIR, "campus")},
    {"name": "Factory", "path": os.path.join(BASE_PRED_DIR, "factory")},
    {"name": "Farm", "path": os.path.join(BASE_PRED_DIR, "farm")},
    {"name": "Gress", "path": os.path.join(BASE_PRED_DIR, "gress")},
]

# 视觉风格配置
CMAP = "RdYlGn"     # 红-黄-绿 (a1指标专用: 绿高红低)

# ================= 2. 数据解析器 (保持不变) =================

def parse_scene_data(folder_path):
    txt_path = os.path.join(folder_path, "Scene_FineGrained_Report.txt")
    summary = {"AbsRel": "-", "a1": "-", "RMSE": "-", "Alt_MAE": "-", "FOV_MAE": "-"}
    heatmap_rows = []
    
    if not os.path.exists(txt_path):
        return summary, pd.DataFrame()

    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    is_combo_section = False
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("OVERALL") and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                try:
                    summary["AbsRel"], summary["RMSE"], summary["a1"] = float(parts[2]), float(parts[3]), float(parts[4])
                except: pass
        if "Height MAE" in line: summary["Alt_MAE"] = line.split(":")[1].strip().split(" ")[0]
        if "FOV MAE" in line: summary["FOV_MAE"] = line.split(":")[1].strip().split(" ")[0]
        if "Pitch | Alt | FOV" in line: is_combo_section = True; continue
        if "Drawing" in line or "Processing" in line: is_combo_section = False
        if is_combo_section and "|" in line and "Pitch" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 7:
                try:
                    heatmap_rows.append({"FOV": parts[0], "Pitch": parts[1], "Altitude": parts[2], "a1": float(parts[6])})
                except: pass
    return summary, pd.DataFrame(heatmap_rows)

# ================= 3. 绘图核心 (支持多FOV) =================

def draw_dashboard():
    # 设置画布: 更宽一些以容纳并排的热力图
    fig = plt.figure(figsize=(28, 18), dpi=150)
    fig.patch.set_facecolor('white')
    
    # 主网格布局
    gs = GridSpec(4, 2, height_ratios=[0.05, 0.12, 0.415, 0.415], hspace=0.3, wspace=0.1)
    
    # --- 1. 标题 & 2. 总表 (保持不变) ---
    ax_title = fig.add_subplot(gs[0, :]); ax_title.axis('off')
    ax_title.text(0.5, 0.5, "🚁 MoGe2 Drone Depth Estimation: Multi-FOV Comprehensive Evaluation", ha='center', va='center', fontsize=24, fontweight='bold', color='#2c3e50')

    ax_table = fig.add_subplot(gs[1, :]); ax_table.axis('off')
    cols = ["Scene", "AbsRel (↓)", "a1 (↑)", "RMSE (↓)", "Alt Err (m)", "FOV Err (°)"]
    cell_data = []
    all_summaries, all_dfs = [], []
    for t in TASKS:
        summ, df = parse_scene_data(t['path'])
        all_summaries.append(summ); all_dfs.append(df)
        def fmt(v, f="{:.4f}"): return f.format(v) if isinstance(v, (int, float)) else v
        cell_data.append([t['name'], fmt(summ['AbsRel']), fmt(summ['a1'], "{:.3f}"), fmt(summ['RMSE'], "{:.2f}"), fmt(summ['Alt_MAE']), fmt(summ['FOV_MAE'])])
        
    table = ax_table.table(cellText=cell_data, colLabels=cols, loc='center', cellLoc='center', bbox=[0.2, 0.0, 0.6, 1.0])
    table.auto_set_font_size(False); table.set_fontsize(13)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_facecolor('#34495e'); cell.set_text_props(color='white', weight='bold')
        elif row > 0 and col == 0: cell.set_text_props(weight='bold')
            
    # --- 3. 绘制四个场景详情 (核心修改区域) ---
    grid_indices = [(2, 0), (2, 1), (3, 0), (3, 1)]
    # 强制排序列表
    order_pitch = ["1. -90°", "2. -75°", "3. -60°", "4. -45°"]
    order_alt = ["1. Low (<50m)", "2. Mid (50-100m)", "3. High (100-200m)", "4. V-High (>200m)"]

    for i, task in enumerate(TASKS):
        # 获取当前场景的大格子
        scene_gs_slot = gs[grid_indices[i]]
        
        summ = all_summaries[i]
        df = all_dfs[i]
        
        # 准备子标题背景
        ax_bg = fig.add_subplot(scene_gs_slot)
        ax_bg.axis('off')
        title_str = f" {task['name']}  |  Overall a1: {summ['a1']}  |  Alt Err: {summ['Alt_MAE']}m "
        ax_bg.text(0.5, 1.05, title_str, ha='center', va='bottom', fontsize=16, fontweight='bold', 
                   color='white', bbox=dict(facecolor='#2980b9', edgecolor='none', pad=5.0))

        if df.empty:
            ax_bg.text(0.5, 0.5, "No Data Available", ha='center', va='center'); continue

        # --- 关键修改：检测所有唯一的 FOV 并创建子网格 ---
        unique_fovs = sorted(df['FOV'].unique())
        num_fovs = len(unique_fovs)
        
        if num_fovs == 0: continue

        # 创建内部嵌套网格 (1行, N列)
        inner_gs = GridSpecFromSubplotSpec(1, num_fovs, subplot_spec=scene_gs_slot, wspace=0.08)

        for fov_idx, fov in enumerate(unique_fovs):
            ax_heatmap = fig.add_subplot(inner_gs[0, fov_idx])
            
            # 筛选当前 FOV 的数据
            df_sub = df[df['FOV'] == fov]
            pivot = df_sub.pivot_table(index='Pitch', columns='Altitude', values='a1', aggfunc='mean')
            
            # 强制排序
            existing_pitch = [p for p in order_pitch if p in pivot.index]
            if existing_pitch: pivot = pivot.reindex(index=existing_pitch)
            else: pivot = pivot.sort_index()
            pivot = pivot.sort_index(axis=1) # Alt 默认字母序即可

            # 绘制热力图
            # 只在最左侧的图显示 Y 轴标签
            show_yaxis = (fov_idx == 0)
            sns.heatmap(pivot, ax=ax_heatmap, annot=True, fmt=".2f", cmap=CMAP, 
                        vmin=0.0, vmax=1.0, linewidths=1, linecolor='white',
                        cbar=True, cbar_kws={"shrink": .7, "label": "a1" if fov_idx == num_fovs-1 else ""},
                        yticklabels=show_yaxis)
            
            # 设置子标题 (FOV 名称)
            ax_heatmap.set_title(f"FOV: {fov}", fontsize=12, fontweight='bold')
            ax_heatmap.set_xlabel("Altitude", fontsize=10)
            if show_yaxis:
                ax_heatmap.set_ylabel("Pitch Angle", fontsize=10, fontweight='bold')
            else:
                ax_heatmap.set_ylabel("") # 隐藏中间图的 Y Label

            ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=0, fontsize=9) 
            plt.setp(ax_heatmap.get_yticklabels(), fontsize=9)

    # 保存
    save_path = os.path.join(BASE_PRED_DIR, "Final_Dense_Report_MultiFOV.png")
    # 增加边距防止标题被裁减
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"🎉 多FOV密集型汇总报告已生成: {save_path}")

if __name__ == "__main__":
    draw_dashboard()