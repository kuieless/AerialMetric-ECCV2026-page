import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# Set global font to Times New Roman
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

# 1. 把四个字体文件的路径放进列表里
font_paths = [
    '/home/szq/moge2/MoGe/moge/scripts/code-final/batch/TIMES.TTF',   # 常规 (Regular)
    '/home/szq/moge2/MoGe/moge/scripts/code-final/batch/TIMESBD.TTF', # 粗体 (Bold)
    '/home/szq/moge2/MoGe/moge/scripts/code-final/batch/TIMESI.TTF',  # 斜体 (Italic)
    '/home/szq/moge2/MoGe/moge/scripts/code-final/batch/TIMESBI.TTF'  # 粗斜体 (Bold Italic)
]

# 2. 循环导入，告诉 Python 这四个文件是一家人
for path in font_paths:
    fm.fontManager.addfont(path)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 它现在能听懂这句话了
plt.rcParams['mathtext.fontset'] = 'stix'

# ==========================================
# 🎨 1. 配色与设置
# ==========================================
bg = { "oblique": "#CCD7E5", "decoupled": "#E0F2E0", "ground": "#F4E8E0" }  # 浅蓝、浅绿、浅红

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

GAP = 0  # 取消上下间隙，合并为一个图
cy = 0   # 中心点设为0
BG_RADIUS = 105

# ==========================================
# 📊 3. 数据定义 (合并为14边形)
# ==========================================
# 合并标签：上半部分 + 下半部分
labels_combined = ['City', 'Natural', 'Rural', 'Building', 'Factory', 'Lawn', 'Farm', 
                   'NYUv2', 'KITTI', 'ETH3D', 'iBims', 'DDAD', 'DIODE', 'HAMMER']

# 14个角度均匀分布在0到2π之间
angles_combined = np.linspace(0, 2 * np.pi, 14, endpoint=False)

# 合并数据：上半部分 + 下半部分
sota_combined = [5.1, 23.0, 0.3, 26.6, 34.8, 19.4, 0.0, 
                 96.7, 64.8, 88.8, 80.5, 73.6, 67.8, 65.3]

zoe_combined = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                91.9, 85.4, 33.7, 67.2, 38.6, 29.3, 3.2]

pro_combined = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                91.9, 38.3, 32.8, 81.5, 35.3, 37.7, 63.0]

uni_combined = [34.1, 73.0, 24.2, 35.0, 46.7, 33.8, 57.9,
                93.4, 91.4, 69.5, 93.1, 77.5, 51.7, 46.8]

metric_combined = [89.5, 73.6, 81.2, 87.6, 90.2, 93.9, 54.2,
                   96.2, 68.6, 83.7, 85.3, 74.2, 77.0, 68.8]

# ==========================================
# 🖌️ 4. 初始化画布
# ==========================================
fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
ax.set_aspect('equal')
ax.axis('off')

def draw_bg_sector(a_start, a_end, color, r=BG_RADIUS):
    t = np.linspace(a_start, a_end, 100)
    x = [0] + (r * np.cos(t)).tolist() + [0]
    y = [0] + (r * np.sin(t)).tolist() + [0]
    ax.fill(x, y, color=color, alpha=1.0, zorder=-10)

# 计算分界线角度：在相邻数据集之间的中点
# Building(索引3)和Rural(索引2)之间的中点
building_rural_mid = (angles_combined[3] + angles_combined[2]) / 2

# Farm(索引6)和NYUv2(索引7)之间的中点
farm_nyuv2_mid = (angles_combined[6] + angles_combined[7]) / 2

# City(索引0, 0°)和HAMMER(索引13, ~334.3°)之间跨越2π边界的中点
# City=0, HAMMER=13*2π/14, 中点需要绕过0°/360°边界
# 正确的中点 = (HAMMER + City + 2π) / 2 = (HAMMER + 2π) / 2 (因为City=0)
city_hammer_mid = (angles_combined[13] + 2 * np.pi) / 2  # ≈ 347.1° ≈ 6.058 rad

# 绘制背景扇形
# 1. AerialMetric Decoupled - 浅绿色 (Building, Factory, Lawn, Farm)
#    从 building_rural_mid 到 farm_nyuv2_mid（连续区间，无跨界）
draw_bg_sector(building_rural_mid, farm_nyuv2_mid, bg["decoupled"])

# 2. Ground Datasets - 浅红色 (NYUv2, KITTI, ETH3D, iBims, DDAD, DIODE, HAMMER)
#    从 farm_nyuv2_mid 到 city_hammer_mid（连续区间，无跨界）
draw_bg_sector(farm_nyuv2_mid, city_hammer_mid, bg["ground"])

# 3. AerialMetric Oblique - 浅蓝色 (City, Natural, Rural)
#    从 city_hammer_mid 到 building_rural_mid，跨越2π/0边界
#    需要分两段绘制：city_hammer_mid -> 2π 和 0 -> building_rural_mid
draw_bg_sector(city_hammer_mid, 2 * np.pi, bg["oblique"])
draw_bg_sector(0, building_rural_mid, bg["oblique"])

# ==========================================
# 🕸️ 5. 网格
# ==========================================
grids = [20, 40, 60, 80, 100]
for r in grids:
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(r * np.cos(t), r * np.sin(t), color='#E0E0E0', ls='--', lw=0.8, zorder=0)
    ax.text(0, r, str(r), ha='center', va='bottom', fontsize=15, color='#A0A0A0')
    ax.text(0, -r, str(r), ha='center', va='top', fontsize=15, color='#A0A0A0')

for a in angles_combined:
    ax.plot([0, BG_RADIUS*np.cos(a)], [0, BG_RADIUS*np.sin(a)], color='#E8E8E8', lw=0.6, zorder=0)

# ==========================================
# ✂️ 6. 多边形 (绘制5个方法)
# ==========================================
def draw_poly(vals, angles, color, fill_alpha, lw, s):
    # 修复多边形闭合：将第一个点重复添加到末尾以闭合多边形
    x = [v * np.cos(a) for v, a in zip(vals, angles)] + [vals[0] * np.cos(angles[0])]
    y = [v * np.sin(a) for v, a in zip(vals, angles)] + [vals[0] * np.sin(angles[0])]
    
    # 绘制多边形轮廓
    ax.plot(x, y, color=color, lw=lw, alpha=0.9, zorder=3)
    # 填充多边形
    ax.fill(x, y, color=color, alpha=fill_alpha, zorder=2)
    # 绘制数据点（不包括闭合点）
    ax.scatter(x[:-1], y[:-1], color=color, s=s, edgecolor='white', lw=1.2, zorder=4)

# 绘制合并后的多边形
draw_poly(sota_combined, angles_combined, color_sota, alpha_sota_fill, 1.5, 30)
draw_poly(zoe_combined, angles_combined, color_zoe, alpha_zoe_fill, 1.5, 30)
draw_poly(pro_combined, angles_combined, color_pro, alpha_pro_fill, 1.5, 30)
draw_poly(uni_combined, angles_combined, color_uni, alpha_uni_fill, 1.5, 30)
draw_poly(metric_combined, angles_combined, color_metric, alpha_metric_fill, 2.5, 35)

# ==========================================
# 📈 7. 【升级版】防重叠数据点标签 (支持任意数量方法)
# ==========================================
placed_labels = []  

def draw_data_labels_multi(methods_data_list, colors_list, angles):
    # 遍历每个角度轴
    for i, a in enumerate(angles):
        # 提取当前轴上所有方法的数据，并与其颜色绑定
        vals_on_axis = [(m_data[i], color) for m_data, color in zip(methods_data_list, colors_list)]
        
        # 将数值排序 (从小到大放置，有效减少拥挤死锁)
        vals_on_axis.sort(key=lambda x: x[0])
        
        for val, color in vals_on_axis:
            r_init = max(val, 10.0)
            
            # 内部避让计算函数
            def get_safe_r(r_target, angle):
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
                        y_out = r_out * np.sin(angle)
                        if not any(np.hypot(x_out - px, y_out - py) < SAFE_DIST for px, py in placed_labels):
                            return r_out
                            
                    if r_in >= MIN_R:
                        x_in = r_in * np.cos(angle)
                        y_in = r_in * np.sin(angle)
                        if not any(np.hypot(x_in - px, y_in - py) < SAFE_DIST for px, py in placed_labels):
                            return r_in
                            
                    r_out += step
                    r_in -= step
                    
                return r_target

            r_final = get_safe_r(r_init, a)
            placed_labels.append((r_final * np.cos(a), r_final * np.sin(a)))

            va = 'center'
            if np.isclose(np.sin(a), 0): 
                va = 'bottom' if np.cos(a) > 0 else 'top'

            # 👇 【关键修改】在这里增加一个向外的径向偏移量
            radial_offset = 8.0 
            r_text = r_final + radial_offset

            # 绘制标签 (使用 r_text 替代原先的 r_final)
            ax.text(r_text * np.cos(a), r_text * np.sin(a), f'{val:.1f}', 
                    ha='center', va=va, size=17, color=color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=5)

# 将5个方法的数据和颜色打包传入
combined_data_all = [sota_combined, zoe_combined, pro_combined, uni_combined, metric_combined]
colors_all = [color_sota, color_zoe, color_pro, color_uni, color_metric]

# 只显示metric的数据标签
draw_data_labels_multi([metric_combined], [color_metric], angles_combined)

# ==========================================
# 🏷️ 8. 边缘标签与标题
# ==========================================
R_LBL = 113
# Color mapping for each dataset label based on its group
label_color_map = {
    'Farm': '#2E7D32', 'Lawn': '#2E7D32', 'Factory': '#2E7D32', 'Building': '#2E7D32',  # AerialMetric Decoupled - green
    'Rural': '#1565C0', 'Natural': '#1565C0', 'City': '#1565C0',                         # AerialMetric Oblique - blue
    'NYUv2': '#B71C1C', 'KITTI': '#B71C1C', 'ETH3D': '#B71C1C', 'iBims': '#B71C1C',     # Ground Datasets - red
    'DDAD': '#B71C1C', 'DIODE': '#B71C1C', 'HAMMER': '#B71C1C',
}
def draw_edge_labels(labels, angles):
    for lbl, a in zip(labels, angles):
        # Building标签往左偏移一点，Factory标签往右偏移一点
        if lbl == 'Building':
            a_draw = a + 0.06
        elif lbl == 'Factory':
            a_draw = a - 0.06
        else:
            a_draw = a
        ha = 'left' if np.cos(a_draw) > 0.01 else ('right' if np.cos(a_draw) < -0.01 else 'center')
        va = 'bottom' if np.sin(a_draw) > 0.01 else ('top' if np.sin(a_draw) < -0.01 else 'center')
        if np.isclose(np.sin(a_draw), 0): va = 'center'
        lbl_color = label_color_map.get(lbl, '#2C3E50')
        ax.text(R_LBL*np.cos(a_draw), R_LBL*np.sin(a_draw), lbl, ha=ha, va=va, fontsize=21, fontweight='bold', color=lbl_color)

draw_edge_labels(labels_combined, angles_combined)

# ==========================================
# 🏷️ 8.5 区域名称标注
# ==========================================
# AerialMetric Decoupled - 浅绿区域 (Building/Factory/Lawn/Farm), 左上方
# 该区域中心角度约在 Lawn(5) 附近
decoupled_center_angle = angles_combined[5]  # Lawn的角度
ax.text(152 * np.cos(decoupled_center_angle + 0.15), 145,
        'AerialMetric\nDecoupled', ha='center', va='center', fontsize=30, fontweight='bold',
        color='#2E7D32', zorder=6)

# AerialMetric Oblique - 浅蓝区域 (Rural/Natural/City), 右上方
# 该区域中心角度约在 Rural(2) 和 Natural(1) 之间
oblique_center_angle = (angles_combined[2] + angles_combined[1]) / 2
ax.text(152 * np.cos(oblique_center_angle), 145,
        'AerialMetric\nOblique', ha='center', va='center', fontsize=30, fontweight='bold',
        color='#1565C0', zorder=6)

# Ground Datasets - 浅红区域 (NYUv2~HAMMER), 下方
# 该区域中心角度约在 ETH3D(9) 和 iBims(10) 之间
ground_center_angle = (angles_combined[10] + angles_combined[11]) / 2
ax.text(0, -135, 'Ground Datasets', ha='center', va='center', fontsize=30, fontweight='bold',
        color='#B71C1C', zorder=6)

# ==========================================
# 🟩 9. 图例 (扩展到5个)
# ==========================================
legend_elements_row1 = [
    Line2D([0], [0], color=color_sota, lw=2.5, label='MoGe2'),
    Line2D([0], [0], color=color_zoe, lw=2.5, label='ZoeDepth'),
    Line2D([0], [0], color=color_pro, lw=2.5, label='DepthPro'),
]
legend_elements_row2 = [
    Line2D([0], [0], color=color_uni, lw=2.5, label='UniDepthV2'),
    Line2D([0], [0], color=color_metric, lw=2.5, label='MoGe2-Aerial'),
]

leg1 = ax.legend(handles=legend_elements_row1, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, prop={'size': 23, 'weight': 'bold'})
# Set legend text colors to match line colors for row1
for text, color in zip(leg1.get_texts(), [color_sota, color_zoe, color_pro]):
    text.set_color(color)
ax.add_artist(leg1)
leg2 = ax.legend(handles=legend_elements_row2, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, prop={'size': 23, 'weight': 'bold'})
# Set legend text colors to match line colors for row2
for text, color in zip(leg2.get_texts(), [color_uni, color_metric]):
    text.set_color(color)

ax.set_xlim(-150, 150)
ax.set_ylim(-155, 155)
plt.savefig('radar_14_sides.pdf', dpi=400, bbox_inches='tight')
plt.savefig('radar_14_sides.png', dpi=600, bbox_inches='tight', facecolor='white')
print("✅ 十四边形雷达图已生成！")
