# #!/bin/bash

# # ================= 📋 任务排队区 =================

# run_all_tasks() {
#     # 格式: run_pipeline "实验名称" "模型路径"
    
#     run_pipeline "Eval-base" \
#                  "/home/szq/moge2/MoGe/vitl-normal.pt"
# }

# # ================= ⚙️ 配置区 =================

# NUM_WORKERS=8
# PYTHON_CMD="python"
# PROJECT_ROOT="/home/szq/moge2/MoGe"
# RAW_DATA_ROOT="/data1/szq/data/eval"
# STAGING_ROOT="/data1/szq/data/benchmark_staging"
# BASE_RESULT_DIR="/data1/szq/data/benchmark_results_v2"

# # --- 脚本路径 (请确保这些文件名正确且在当前目录下) ---
# SCRIPT_STEP0="${PWD}/eva-0-preprocess_data.py"
# SCRIPT_STEP3="${PWD}/eva-c-real-tasks-pitch-altitude-fov-syn-debug-batch-classes4 copy.py"

# # --- 你的原始脚本路径 ---
# SCRIPT_STEP1="/home/szq/moge2/MoGe/moge/scripts/eva-a-infer_batch_Real-final-batch.py"
# SCRIPT_STEP2="/home/szq/moge2/MoGe/moge/scripts/b-ex-infer-ex-Real.py"

# # ================= 🧠 核心逻辑 =================

# # 强制重置 Step 0 (为了修复你刚才的目录结构问题)
# force_reset_step0() {
#     echo "🧹 强制清理旧的 Staging 数据，以适配新的目录结构..."
#     rm -f "${STAGING_ROOT}/benchmark_index.csv"
#     rm -rf "${STAGING_ROOT}/images"
# }

# check_and_run_step0() {
#     CSV_PATH="${STAGING_ROOT}/benchmark_index.csv"
#     # Step 0 会在 output_root/images/mixed_val 下生成图片
#     # Step 1 需要的输入是 output_root/images
#     INPUT_FOR_STEP1="${STAGING_ROOT}/images" 

#     if [ ! -f "$CSV_PATH" ]; then
#         echo "🔄 运行 Step 0 (数据预处理)..."
#         $PYTHON_CMD $SCRIPT_STEP0 \
#             --source_root "$RAW_DATA_ROOT" \
#             --output_root "$STAGING_ROOT" \
#             --sample_rate 0.15 \
#             || { echo "❌ Step 0 失败"; exit 1; }
#     fi
#     echo "$INPUT_FOR_STEP1"
# }

# run_pipeline() {
#     local EXP_NAME=$1
#     local MODEL_PATH=$2
    
#     local STEP1_INPUT_DIR=$(check_and_run_step0) # 获取 Step 0 的输出目录
#     local STEP1_OUT="${BASE_RESULT_DIR}/${EXP_NAME}"
#     local STEP2_OUT="${BASE_RESULT_DIR}/${EXP_NAME}-out"
#     local LOG_DIR="${BASE_RESULT_DIR}/logs_${EXP_NAME}"
#     local CSV_PATH="${STAGING_ROOT}/benchmark_index.csv"
    
#     mkdir -p "$STEP1_OUT" "$STEP2_OUT" "$LOG_DIR"

#     echo "############################################################"
#     echo "🟢 [任务] ${EXP_NAME}"
#     echo "📂 [输入] ${STEP1_INPUT_DIR} (包含子文件夹)"
#     echo "############################################################"

#     set -e 

#     # --- Step 1 ---
#     echo -e "\n▶️ [1/3] Step 1: 推理..."
    
#     for ((i=0; i<NUM_WORKERS; i++)); do
#         $PYTHON_CMD $SCRIPT_STEP1 \
#             --model_path "$MODEL_PATH" \
#             --input_root "$STEP1_INPUT_DIR" \
#             --output_root "$STEP1_OUT" \
#             --project_root "$PROJECT_ROOT" \
#             --total_shards "$NUM_WORKERS" \
#             --shard_id "$i" \
#             > "${LOG_DIR}/worker_${i}.log" 2>&1 &
#     done
#     wait

#     # 🚨 错误检查：如果 Step 1 生成的文件夹是空的，打印日志 🚨
#     if [ -z "$(ls -A $STEP1_OUT)" ]; then
#         echo "❌ [Error] Step 1 完成但没有生成任何输出！"
#         echo "👇 查看 worker_0.log 的报错信息："
#         echo "------------------------------------------------"
#         cat "${LOG_DIR}/worker_0.log" | tail -n 20
#         echo "------------------------------------------------"
#         exit 1
#     fi
#     echo "✅ Step 1 完成。"

#     # --- Step 2 ---
#     echo -e "\n▶️ [2/3] Step 2: 提取..."
#     $PYTHON_CMD $SCRIPT_STEP2 \
#         --input_dir "$STEP1_OUT" \
#         --output_dir "$STEP2_OUT" \
#         > "${LOG_DIR}/step2.log" 2>&1
#     echo "✅ Step 2 完成。"

#     # --- Step 3 ---
#     echo -e "\n▶️ [3/3] Step 3: 评估..."
#     if [ ! -f "$SCRIPT_STEP3" ]; then
#         echo "❌ 找不到评估脚本: $SCRIPT_STEP3"
#         echo "请确保 3-eval_benchmark.py 在当前目录！"
#         exit 1
#     fi
    
#     $PYTHON_CMD $SCRIPT_STEP3 \
#         --pred_root "$STEP2_OUT" \
#         --csv_path "$CSV_PATH" \
#         --save_report "${BASE_RESULT_DIR}/${EXP_NAME}_report.txt" \
#         2>&1 | tee "${LOG_DIR}/step3.log"
        
#     echo -e "\n🎉 报告: ${BASE_RESULT_DIR}/${EXP_NAME}_report.txt"
#     sleep 3
# }

# # ================= 🚀 启动 =================

# # 1. 第一次运行前，强制清除旧的错误格式数据
# if [ ! -d "${STAGING_ROOT}/images/mixed_val" ]; then
#     force_reset_step0
# fi

# run_all_tasks

#!/bin/bash

# ================= 📋 任务配置区 =================

# 1. 实验名称 (Output 文件夹名字)
EXP_NAME="Benchmark-Eval-v5"

# 2. 模型路径
MODEL_PATH="/home/szq/moge2/MoGe/vitl-normal.pt"
# MODEL_PATH="/home/szq/moge2/MoGe/workspace/final-fintune2-1.18-multiloss-patch8-normal3/checkpoint/00001500.pt"

# 3. 路径配置
PROJECT_ROOT="/home/szq/moge2/MoGe"
RAW_DATA_ROOT="/data1/szq/data/eval"               # 原始数据集根目录
STAGING_ROOT="/data1/szq/data/benchmark_staging"   # Step 0 预处理输出目录
BASE_RESULT_DIR="/data1/szq/data/benchmark_results" # 最终结果目录

# 4. Python 命令与环境
PYTHON_CMD="python"  # 如果需要指定 conda 环境，可以用 /path/to/conda/bin/python
NUM_WORKERS=8        # 并行进程数

# ================= 🔍 脚本定位 =================
# 假设这4个脚本都在当前目录下，如果不是，请修改路径
SCRIPT_S0="eva-0-preprocess_data.py"
SCRIPT_S1="eva-a-infer_batch_Real-final-batch.py"
SCRIPT_S2="eval-b-ex-infer-ex-Real.py"
SCRIPT_S3="eva-c-real-tasks-pitch-altitude-fov-syn-debug-batch-classes4.py"

# 检查脚本是否存在
for script in $SCRIPT_S0 $SCRIPT_S1 $SCRIPT_S2 $SCRIPT_S3; do
    if [ ! -f "$script" ]; then
        echo "❌ 错误: 找不到脚本文件 '$script'"
        echo "   请确保所有 python 脚本都在当前目录下，或者修改 shell 脚本中的路径。"
        exit 1
    fi
done

# ================= 🚀 核心流程 =================

# 准备日志目录
mkdir -p "$BASE_RESULT_DIR"
LOG_DIR="${BASE_RESULT_DIR}/${EXP_NAME}_logs"
mkdir -p "$LOG_DIR"

echo "############################################################"
echo "🟢 [Start] 任务: ${EXP_NAME}"
echo "📦 [Model] ${MODEL_PATH}"
echo "🕒 [Time] $(date)"
echo "############################################################"

set -e # 遇到错误立即停止

# --- Step 0: 预处理 ---
echo -e "\n▶️ [0/3] Step 0: 数据预处理..."
CSV_PATH="${STAGING_ROOT}/benchmark_index.csv"

# 检查是否需要重新运行 Step 0
# 如果 CSV 不存在，或者 staging/images/mixed_val 不存在，则重新运行
if [ ! -f "$CSV_PATH" ] || [ ! -d "${STAGING_ROOT}/images/mixed_val" ]; then
    echo "   正在生成索引和软链接..."
    $PYTHON_CMD $SCRIPT_S0 \
        --source_root "$RAW_DATA_ROOT" \
        --output_root "$STAGING_ROOT" \
        --all \
        > "${LOG_DIR}/step0.log" 2>&1
    echo "✅ Step 0 完成。"
else
    echo "✅ Step 0 已存在，跳过预处理 (如需重跑请删除 $STAGING_ROOT)。"
fi

# --sample_rate 0.2

# --- Step 1: 推理 ---
# Step 1 输出目录
STEP1_OUT="${BASE_RESULT_DIR}/${EXP_NAME}/raw_preds"
# Step 0 生成的图片目录 (input_root)
INPUT_IMGS_DIR="${STAGING_ROOT}/images"

echo -e "\n▶️ [1/3] Step 1: 并行推理 (Image Sharding)..."
echo "   输入: $INPUT_IMGS_DIR"
echo "   输出: $STEP1_OUT"

# 启动多进程
pids=""
for ((i=0; i<NUM_WORKERS; i++)); do
    $PYTHON_CMD $SCRIPT_S1 \
        --model_path "$MODEL_PATH" \
        --input_root "$INPUT_IMGS_DIR" \
        --output_root "$STEP1_OUT" \
        --project_root "$PROJECT_ROOT" \
        --total_shards "$NUM_WORKERS" \
        --shard_id "$i" \
        > "${LOG_DIR}/worker_${i}.log" 2>&1 &
    
    pids="$pids $!"
done

# 等待所有子进程
wait $pids
echo "✅ Step 1 推理完成。"


# --- Step 2: 提取 ---
STEP2_OUT="${BASE_RESULT_DIR}/${EXP_NAME}/npy_flat"

echo -e "\n▶️ [2/3] Step 2: 结果扁平化提取..."
$PYTHON_CMD $SCRIPT_S2 \
    --input_dir "$STEP1_OUT" \
    --output_dir "$STEP2_OUT" \
    > "${LOG_DIR}/step2.log" 2>&1

echo "✅ Step 2 提取完成。"


# --- Step 3: 评估 ---
FINAL_RESULT_DIR="${BASE_RESULT_DIR}/${EXP_NAME}"

echo -e "\n▶️ [3/3] Step 3: 16-bit PNG 评估..."
$PYTHON_CMD $SCRIPT_S3 \
    --pred_root "$STEP2_OUT" \
    --csv_path "$CSV_PATH" \
    --output_dir "$FINAL_RESULT_DIR" \
    2>&1 | tee "${LOG_DIR}/step3.log"

echo -e "\n🎉 任务全部结束！"
echo "📄 评估报告: ${FINAL_RESULT_DIR}/Final_Report.txt"