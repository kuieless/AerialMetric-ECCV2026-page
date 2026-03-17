#!/bin/bash

# ================= 配置区域 =================

# Python 脚本绝对路径
PY_SCRIPT="/path/to/your/auto_upsample_depth.py"

# 插值方法: bicubic (推荐) 或 bilinear
METHOD="bicubic"

# 任务列表
# 格式： "预测结果目录 | 参考真值目录 | 输出目录"
# 逻辑： 把 '预测结果' 缩放到 '参考真值' 的大小，存入 '输出'
TASKS=(
    "/data/res/scene_01/pred_small|/data/raw/scene_01/gt|/data/res/scene_01/pred_upsampled"
    "/data/res/scene_02/pred_small|/data/raw/scene_02/images|/data/res/scene_02/pred_upsampled"
)

# ================= 执行逻辑 =================

echo "==========================================="
echo "Batch Depth Upsampling Task"
echo "Method: $METHOD"
echo "==========================================="

for entry in "${TASKS[@]}"; do
    IFS='|' read -r pred_dir gt_dir output_dir <<< "$entry"
    
    # 清理空格
    pred_dir=$(echo "$pred_dir" | xargs)
    gt_dir=$(echo "$gt_dir" | xargs)
    output_dir=$(echo "$output_dir" | xargs)

    echo ""
    echo ">>> Processing..."
    echo "    Pred Source: $pred_dir"
    echo "    Ref Size From: $gt_dir"
    echo "    Output To:   $output_dir"

    # 检查目录
    if [ ! -d "$pred_dir" ]; then
        echo "❌ Pred dir not found: $pred_dir"
        continue
    fi
    if [ ! -d "$gt_dir" ]; then
        echo "❌ GT/Ref dir not found: $gt_dir"
        continue
    fi

    # 调用 Python
    python "$PY_SCRIPT" \
        --pred "$pred_dir" \
        --gt "$gt_dir" \
        --output "$output_dir" \
        --method "$METHOD"

    if [ $? -eq 0 ]; then
        echo "✅ Upsample Finished."
    else
        echo "❌ Upsample Failed."
    fi
done