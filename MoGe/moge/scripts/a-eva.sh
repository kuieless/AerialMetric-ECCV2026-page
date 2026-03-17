#!/bin/bash

# ================= 🔧 你的配置 (只改这里) =================

# 1. 原始数据在哪里？(里面包含 DDAD/val/...)
RAW_DATA_ROOT="/home/data1/szq/Megadepth/Benchmark-final2/moge-eva"

# 2. 你想把抽出来的 20% 放在哪？(这是个临时文件夹，会自动创建)
WORK_DIR="/home/data1/szq/Megadepth/Benchmark-final2/moge-eva-Mini-Test-Results"

# 3. 模型权重路径
# MODEL_PATH="/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/workspace/final-fintune-scalehead/checkpoint/00001000.pt"
MODEL_PATH="/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt"

# ================= 🚀 自动流程 (请勿修改) =================
set -e # 报错即停止

echo "========================================"
echo "🚀 开始执行: 采样 -> 推理 -> 评估"
echo "========================================"

# --- 1. 采样 & 准备数据 ---
if [ ! -d "$WORK_DIR" ] || [ -z "$(ls -A $WORK_DIR)" ]; then
    echo -e "\n📦 [Step 1] 正在抽取 20% 数据..."
    python prepare_subset.py \
        --src_root "$RAW_DATA_ROOT" \
        --dst_root "$WORK_DIR" \
        --ratio 0.2
else
    echo -e "\n⏭️ [Step 1] 目录已存在，跳过采样。"
fi

# --- 2. 推理 ---
echo -e "\n🧠 [Step 2] 开始模型推理..."
# 我们直接把结果输出到 WORK_DIR 里面，这样 depth.npy 会和 depth_gt.png 在一起
# 注意：前提是你的 step1_inference.py 逻辑是把结果存到 output_root/scene_name/
# 为了稳妥，我们把 output 设为 WORK_DIR 的一个子目录 results，然后把 GT 拷过去
RESULT_DIR="${WORK_DIR}_Results"

python /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/moge/scripts/a-infer_batch_Real-final.py \
    --model_path "$MODEL_PATH" \
    --input_root "$WORK_DIR" \
    --output_root "$RESULT_DIR" \
    --resize 1024

# --- 3. 搬运 GT (关键一步) ---
# step1 跑完后，RESULT_DIR 里只有 image.jpg 和 depth.npy (及其他json)
# 但 step3 评估需要 depth_gt.png。
# 我们把 WORK_DIR 里的 depth_gt.png 复制到 RESULT_DIR 对应的文件夹里。
echo -e "\n🚚 [Step 2.5] 正在搬运 GT 文件以供评估..."

# 进入源目录，找到所有 png，保持结构复制到结果目录
cd "$WORK_DIR"
find . -name "depth_gt.png" | cpio -pdm "$RESULT_DIR" > /dev/null 2>&1
cd -

# --- 4. 评估 ---
echo -e "\n📈 [Step 3] 开始最终评估..."

python step3_eval_png.py \
    --pred_root "$RESULT_DIR"

echo -e "\n✅ 全部完成！结果在上面。"