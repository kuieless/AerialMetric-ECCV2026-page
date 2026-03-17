
START_STAGE=1
EXP_NAME="Val-Results-moge2-1000step-neck"

# 1. 路径定义
MODEL_PATH="/home/szq/moge2/MoGe/workspace/final-fintune-scalehead-neck/checkpoint/00001000.pt"
DATASET_INPUT="/data1/szq/data/Val"
GT_ROOT="/data1/szq/data/Val"
CSV_PATH="/home/szq/moge2/final_merged.csv"
BASE_RESULT_DIR="/data1/szq/data/becnmarkdata2"

# 🔥 新增：明确指定项目根目录 (包含 moge 文件夹的那个父目录)
PROJECT_ROOT="/home/szq/moge2/MoGe" 

# ================= ⚙️ 自动路径生成 =================

STEP1_OUT="${BASE_RESULT_DIR}/${EXP_NAME}"
STEP2_OUT="${BASE_RESULT_DIR}/${EXP_NAME}-out"
LOG_FILE="${BASE_RESULT_DIR}/log_${EXP_NAME}.txt"
mkdir -p "$BASE_RESULT_DIR"

# ================= 🚀 执行流程 =================

set -e
PYTHON_CMD="python"

echo "========================================================" | tee -a "$LOG_FILE"
echo "🧪 实验名称: $EXP_NAME" | tee -a "$LOG_FILE"
echo "🕒 时间: $(date)" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

# --- Step 1: 推理 ---
if [ "$START_STAGE" -le 1 ]; then
    echo -e "\n▶️ [1/3] 开始推理..." | tee -a "$LOG_FILE"
    
    $PYTHON_CMD /home/szq/moge2/MoGe/moge/scripts/a-infer_batch_Real-final.py \
        --model_path "$MODEL_PATH" \
        --input_root "$DATASET_INPUT" \
        --output_root "$STEP1_OUT" \
        --project_root "$PROJECT_ROOT" \
        2>&1 | tee -a "$LOG_FILE"
        # 👆 上面这一行是关键修复，显式传入路径
else
    echo -e "\n⏭️ [1/3] 跳过推理..." | tee -a "$LOG_FILE"
fi

# --- Step 2: 提取 ---
if [ "$START_STAGE" -le 2 ]; then
    echo -e "\n▶️ [2/3] 开始提取 NPY..." | tee -a "$LOG_FILE"
    
    # 注意：如果这个脚本也依赖 moge 包，可能也需要加 --project_root
    $PYTHON_CMD /home/szq/moge2/MoGe/moge/scripts/b-ex-infer-ex-Real.py \
        --input_dir "$STEP1_OUT" \
        --output_dir "$STEP2_OUT" \
        2>&1 | tee -a "$LOG_FILE"
else
    echo -e "\n⏭️ [2/3] 跳过提取..." | tee -a "$LOG_FILE"
fi

# --- Step 3: 评估 ---
if [ "$START_STAGE" -le 3 ]; then
    echo -e "\n▶️ [3/3] 开始评估..." | tee -a "$LOG_FILE"

    $PYTHON_CMD /home/szq/moge2/MoGe/moge/scripts/c-real-tasks-pitch-altitude-fov-syn-debug-batch-classes4.py \
        --pred_root "$STEP2_OUT" \
        --gt_root "$GT_ROOT" \
        --csv_path "$CSV_PATH" \
        2>&1 | tee -a "$LOG_FILE"
fi

echo -e "\n✅ 流程结束！" | tee -a "$LOG_FILE"