#!/bin/bash

# ================= 配置区域 =================

# Python 脚本绝对路径
PY_SCRIPT="/path/to/your/auto_resize.py"

# 下采样参数
TARGET_WIDTH=640   # 你希望缩放到的宽度
ALIGN_BASE=14      # 保证缩放后的长宽依然是14的倍数

# 任务列表 "输入|输出"
TASKS=(
    "/data/raw/scene_01/images|/data/processed/scene_01/images_small"
    "/data/raw/scene_02/images|/data/processed/scene_02/images_small"
)

# ================= 执行逻辑 =================

echo "==========================================="
echo "Batch Downsampling Task"
echo "Target Width: ~$TARGET_WIDTH"
echo "Force Align:  Multiple of $ALIGN_BASE"
echo "==========================================="

for entry in "${TASKS[@]}"; do
    IFS='|' read -r input_dir output_dir <<< "$entry"
    input_dir=$(echo "$input_dir" | xargs)
    output_dir=$(echo "$output_dir" | xargs)

    echo ""
    echo ">>> Processing..."
    echo "    In:  $input_dir"
    echo "    Out: $output_dir"

    if [ ! -d "$input_dir" ]; then
        echo "❌ Input dir not found, skipping."
        continue
    fi

    # 调用 Python
    python "$PY_SCRIPT" \
        --input "$input_dir" \
        --output "$output_dir" \
        --width "$TARGET_WIDTH" \
        --align "$ALIGN_BASE"

    if [ $? -eq 0 ]; then
        echo "✅ Done."
    else
        echo "❌ Failed."
    fi
done