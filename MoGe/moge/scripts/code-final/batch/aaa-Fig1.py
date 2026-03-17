
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ==========================================
# 🎨 1. 配色与设置
# ==========================================
bg = { "oblique": "#CCD7E5", "decoupled": "#E8EEE7", "ground": "#F4E8E0" }

color_sota = '#7F8C8D'        # 灰色
color_zoe = "#33B86A"         # 绿色
color_pro = '#3498DB'         # 蓝色
color_uni = '#9B59B6'         # 紫色
color_metric = '#C0392B'      # 红色 (重点突出)

# 统一透明度
alpha_sota_fill = 0.04        
alpha_zoe_fill = 0.04
alpha_pro_fill = 0.04
alpha_uni_fill = 0.04
alpha_metric_fill = 0.15      

GAP = 40  
cy_top, cy_bot = GAP/2, -GAP/2
BG_RADIUS = 105

# ==========================================
# 📊 3. 数据定义 (Top: Aerial)
# ==========================================
labels_top_ordered = ['City', 'Natural', 'Rural', 'Building', 'Factory',  'Lawn','Farm']
angles_top_ordered = np.linspace(0, np.pi, 7)

sota_top_final   = [5.1, 23.0, 0.3, 26.6, 34.8,  19.4,0.0] 
zoe_top_final    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
pro_top_final    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
uni_top_final    = [34.1, 73.0, 24.2, 35.0, 46.7, 33.8,57.9 ]
metric_top_final = [89.5, 73.6, 81.2, 87.6, 90.2, 93.9,54.2, ] 

# ==========================================
# 📊 3. 数据定义 (Bottom: Ground) 
# ==========================================
labels_bot_ordered = ['NYUv2', 'KITTI', 'ETH3D', 'iBims', 'DDAD', 'DIODE', 'HAMMER']
angles_bot_ordered = np.linspace(np.pi, 2 * np.pi, 7) 

sota_bot_final   = [96.7, 64.8, 88.8, 80.5, 73.6, 67.8, 65.3] 
# 补充三个新方法的占位数据 (请替换为你真实的实验数据)
zoe_bot_final    = [91.9, 85.4, 33.7, 67.2, 38.6, 29.3, 3.2]
pro_bot_final    = [91.9, 38.3, 32.8, 81.5, 35.3, 37.7, 63.0]
uni_bot_final    = [93.4, 91.4, 69.5, 93.1, 77.5, 51.7, 46.8]
metric_bot_final = [96.2, 68.6, 83.7, 85.3, 74.2, 77.0, 68.8] 

# ==========================================
# 🖌️ 4. 初始化画布
# ==========================================
fig, ax = plt.subplots(figsize=(12, 14), facecolor='white')
ax.set_aspect('equal')
ax.axis('off')

def draw_bg_slice(a_start, a_end, cy, color, r=BG_RADIUS):
    t = np.linspace(a_start, a_end, 100)
    x = [0] + (r * np.cos(t)).tolist() + [0]
    y = [cy] + (r * np.sin(t) + cy).tolist() + [cy]
    ax.fill(x, y, color=color, alpha=1.0, zorder=-10)

draw_bg_slice(0, 0.5 * np.pi, cy_top, bg["oblique"])    
draw_bg_slice(0.5 * np.pi, np.pi, cy_top, bg["decoupled"]) 
draw_bg_slice(np.pi, 2 * np.pi, cy_bot, bg["ground"])   

# ==========================================
# 🕸️ 5. 网格
# ==========================================
grids = [20, 40, 60, 80, 100]
for r in grids:
    for cy, a_range in [(cy_top, (0, np.pi)), (cy_bot, (np.pi, 2*np.pi))]:
        t = np.linspace(a_range[0], a_range[1], 100)
        ax.plot(r * np.cos(t), r * np.sin(t) + cy, color='#E0E0E0', ls='--', lw=0.8, zorder=0)
        ax.plot([r * np.cos(a_range[0]), r * np.cos(a_range[1])], [cy, cy], color='#E0E0E0', ls='-', lw=0.6, zorder=0)
        ax.text(0, (r+cy if cy>0 else -r+cy), str(r), ha='center', va='bottom' if cy>0 else 'top', fontsize=10, color='#A0A0A0')

for a in angles_top_ordered: ax.plot([0, BG_RADIUS*np.cos(a)], [cy_top, BG_RADIUS*np.sin(a)+cy_top], color='#E8E8E8', lw=0.6, zorder=0)
for a in angles_bot_ordered: ax.plot([0, BG_RADIUS*np.cos(a)], [cy_bot, BG_RADIUS*np.sin(a)+cy_bot], color='#E8E8E8', lw=0.6, zorder=0)

# ==========================================
# ✂️ 6. 多边形 (绘制5个方法)
# ==========================================
def draw_poly(vals, angles, cy, color, fill_alpha, lw, s):
    x = [0] + [v * np.cos(a) for v, a in zip(vals, angles)] + [0]
    y = [cy] + [v * np.sin(a) + cy for v, a in zip(vals, angles)] + [cy]
    ax.plot(x, y, color=color, lw=lw, alpha=0.9, zorder=3)
    ax.fill(x, y, color=color, alpha=fill_alpha, zorder=2)
    ax.scatter(x[1:-1], y[1:-1], color=color, s=s, edgecolor='white', lw=1.2, zorder=4)

# Top
draw_poly(sota_top_final, angles_top_ordered, cy_top, color_sota, alpha_sota_fill, 1.5, 30)
draw_poly(zoe_top_final, angles_top_ordered, cy_top, color_zoe, alpha_zoe_fill, 1.5, 30)
draw_poly(pro_top_final, angles_top_ordered, cy_top, color_pro, alpha_pro_fill, 1.5, 30)
draw_poly(uni_top_final, angles_top_ordered, cy_top, color_uni, alpha_uni_fill, 1.5, 30)
draw_poly(metric_top_final, angles_top_ordered, cy_top, color_metric, alpha_metric_fill, 2.5, 35)

# Bottom
draw_poly(sota_bot_final, angles_bot_ordered, cy_bot, color_sota, alpha_sota_fill, 1.5, 30)
draw_poly(zoe_bot_final, angles_bot_ordered, cy_bot, color_zoe, alpha_zoe_fill, 1.5, 30)
draw_poly(pro_bot_final, angles_bot_ordered, cy_bot, color_pro, alpha_pro_fill, 1.5, 30)
draw_poly(uni_bot_final, angles_bot_ordered, cy_bot, color_uni, alpha_uni_fill, 1.5, 30)
draw_poly(metric_bot_final, angles_bot_ordered, cy_bot, color_metric, alpha_metric_fill, 2.5, 35)


# ==========================================
# 📈 7. 【升级版】防重叠数据点标签 (支持任意数量方法)
# ==========================================
placed_labels = []  

def draw_data_labels_multi(methods_data_list, colors_list, angles, cy):
    # 遍历每个角度轴
    for i, a in enumerate(angles):
        # 提取当前轴上所有方法的数据，并与其颜色绑定
        vals_on_axis = [(m_data[i], color) for m_data, color in zip(methods_data_list, colors_list)]
        
        # 将数值排序 (从小到大放置，有效减少拥挤死锁)
        vals_on_axis.sort(key=lambda x: x[0])
        
        for val, color in vals_on_axis:
            r_init = max(val, 10.0)
            
            # 内部避让计算函数
            def get_safe_r(r_target, angle, cy_offset):
                MAX_R = 108.0    
                MIN_R = 10.0     
                # 【重要】为了放下5个标签，安全距离适当缩小为 12.0
                SAFE_DIST = 12.0 
                
                r_target = max(MIN_R, min(r_target, MAX_R))
                r_out, r_in = r_target, r_target
                step = 1.0 
                
                for _ in range(80): 
                    if r_out <= MAX_R:
                        x_out = r_out * np.cos(angle)
                        y_out = r_out * np.sin(angle) + cy_offset
                        if not any(np.hypot(x_out - px, y_out - py) < SAFE_DIST for px, py in placed_labels):
                            return r_out
                            
                    if r_in >= MIN_R:
                        x_in = r_in * np.cos(angle)
                        y_in = r_in * np.sin(angle) + cy_offset
                        if not any(np.hypot(x_in - px, y_in - py) < SAFE_DIST for px, py in placed_labels):
                            return r_in
                            
                    r_out += step
                    r_in -= step
                    
                return r_target

            r_final = get_safe_r(r_init, a, cy)
            placed_labels.append((r_final * np.cos(a), r_final * np.sin(a) + cy))

            va = 'center'
            if np.isclose(np.sin(a), 0): 
                va = 'bottom' if cy > 0 else 'top'

            # 绘制标签 (字体大小从 15 调低至 12 以防止相互覆盖)
            va = 'center'
            if np.isclose(np.sin(a), 0): 
                va = 'bottom' if cy > 0 else 'top'

            # 👇 【关键修改】在这里增加一个向外的径向偏移量
            # 数值越大，数字离中心越远。建议设置在 6.0 到 10.0 之间
            radial_offset = 8.0 
            r_text = r_final + radial_offset

            # 绘制标签 (使用 r_text 替代原先的 r_final)
            ax.text(r_text * np.cos(a), r_text * np.sin(a) + cy, f'{val:.1f}', 
                    ha='center', va=va, size=12, color=color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=5)

# 将5个方法的数据和颜色打包传入
top_data_all = [sota_top_final, zoe_top_final, pro_top_final, uni_top_final, metric_top_final]
bot_data_all = [sota_bot_final, zoe_bot_final, pro_bot_final, uni_bot_final, metric_bot_final]
colors_all = [color_sota, color_zoe, color_pro, color_uni, color_metric]

# draw_data_labels_multi(top_data_all, colors_all, angles_top_ordered, cy_top)
# draw_data_labels_multi(bot_data_all, colors_all, angles_bot_ordered, cy_bot)
draw_data_labels_multi([metric_top_final], [color_metric], angles_top_ordered, cy_top)
draw_data_labels_multi([metric_bot_final], [color_metric], angles_bot_ordered, cy_bot)
# ==========================================
# 🏷️ 8. 边缘标签与标题
# ==========================================
R_LBL = 118 
def draw_edge_labels(labels, angles, cy):
    for lbl, a in zip(labels, angles):
        ha = 'left' if np.cos(a) > 0.01 else ('right' if np.cos(a) < -0.01 else 'center')
        va = 'bottom' if np.sin(a) > 0.01 else ('top' if np.sin(a) < -0.01 else 'center')
        if np.isclose(np.sin(a), 0): va = 'center'
        ax.text(R_LBL*np.cos(a), R_LBL*np.sin(a)+cy, lbl, ha=ha, va=va, fontsize=12, fontweight='bold', color='#2C3E50')

draw_edge_labels(labels_top_ordered, angles_top_ordered, cy_top)
draw_edge_labels(labels_bot_ordered, angles_bot_ordered, cy_bot)

ax.text(55, 125 + cy_top, "Aerial (Oblique)", ha='center', va='center', fontsize=14, fontweight='bold', color='#34495E')
ax.text(-55, 125 + cy_top, "Aerial (Decoupled)", ha='center', va='center', fontsize=14, fontweight='bold', color='#34495E')
ax.text(0, -138 + cy_bot, "Ground Datasets", ha='center', va='center', fontsize=14, fontweight='bold', color='#34495E')

# ==========================================
# 🟩 9. 图例 (扩展到5个)
# ==========================================
legend_elements = [
    Line2D([0], [0], color=color_sota, lw=2.5, label='SoTA'),
    Line2D([0], [0], color=color_zoe, lw=2.5, label='Zoe'),
    Line2D([0], [0], color=color_pro, lw=2.5, label='Pro'),
    Line2D([0], [0], color=color_uni, lw=2.5, label='Uni'),
    Line2D([0], [0], color=color_metric, lw=2.5, label='MetricAnything')
]

# ncol 设置为 5 将一行显示完毕，调整 size 让它们排得下
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False, prop={'size': 13, 'weight': 'bold'})

ax.set_xlim(-150, 150)
ax.set_ylim(-155 + cy_bot, 155 + cy_top)
plt.savefig('radar_teaser_5_methods.png', dpi=400, bbox_inches='tight')
print("✅ 5个方法防重叠版雷达图已生成！")
