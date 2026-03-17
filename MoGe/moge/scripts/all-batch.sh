#!/bin/bash

# ================= 📋 任务排队区 (在这里设置任务!) =================

# 定义一个函数来添加任务
# 格式: run_pipeline "实验名称" "模型路径"
# 您可以在这里无限添加任务...

run_all_tasks() {

    # # === 任务 1 ===
    # run_pipeline "Val-Results-moge2-all-122-origin-loss-4k" \
    #              "/home/szq/moge2/MoGe/workspace/final-fintune2-1.18-multiloss-patch8-normal4-122-4/checkpoint/00004000_ema.pt"


    # run_pipeline "Val-Results-moge2-all-122-origin-loss-8k" \
    #              "/home/szq/moge2/MoGe/workspace/final-fintune2-1.18-multiloss-patch8-normal4-122-4/checkpoint/00008000_ema.pt"

    # run_pipeline "Val-Results-moge2-all-122-origin-loss-11k" \
    #              "/home/szq/moge2/MoGe/workspace/final-fintune2-1.18-multiloss-patch8-normal4-122-4/checkpoint/00011000_ema.pt"





    # # # === 任务 2 (示例) ===
    # run_pipeline "Val-Results-moge2-500step-neck" \
    #              "/home/szq/moge2/MoGe/workspace/final-fintune-scalehead-neck121-3/checkpoint/00000500.pt"

    # run_pipeline "Val-Results-moge2-neck-2k" \
    #              "/home/szq/moge2/MoGe/workspace/final-fintune-scalehead-neck123/checkpoint/00002000_ema.pt"
    # run_pipeline "Val-Results-moge2-neck-4k" \
    #              "/home/szq/moge2/MoGe/workspace/final-fintune-scalehead-neck123/checkpoint/00004000_ema.pt"
    # run_pipeline "Val-Results-moge2-neck-6k" \
    #     "/home/szq/moge2/MoGe/workspace/final-fintune-scalehead-neck123/checkpoint/00006000_ema.pt"

    # run_pipeline "Val-Results-moge2-neck-8k" \
    #     "/home/szq/moge2/MoGe/workspace/final-fintune-scalehead-neck123/checkpoint/00008000_ema.pt"
  
    # run_pipeline "Val-Results-moge2-neck-10k" \
    #     "/home/szq/moge2/MoGe/workspace/final-fintune-scalehead-neck123/checkpoint/00010000_ema.pt"

    # run_pipeline "Val-Results-moge2-neck-12k" \
    #     "/home/szq/moge2/MoGe/workspace/final-fintune-scalehead-neck123/checkpoint/00012000_ema.pt"
    # run_pipeline "Val-Results-moge2-neck-13k" \
    #     "/home/szq/moge2/MoGe/workspace/final-fintune-scalehead-neck123/checkpoint/00013000_ema.pt"

#  CUDA_VISIBLE_DEVICES=6
    # # === Head ===



    run_pipeline "Val-Results-moge2-head-15k" \
                 "/home/szq/moge2/MoGe/workspace/final-fintune-scalehead-head124/checkpoint/00001500_ema.pt"


}

# ================= ⚙️ 全局配置 (通常不用变) =================

NUM_WORKERS=12                 # 并行进程数
DATASET_INPUT="/data1/szq/data/Val"
GT_ROOT="/data1/szq/data/Val"
CSV_PATH="/home/szq/moge2/final_merged.csv"
BASE_RESULT_DIR="/data1/szq/data/becnmarkdata2"
PROJECT_ROOT="/home/szq/moge2/MoGe"
PYTHON_CMD="python"

# ================= 🧠 核心逻辑函数 (不要动这里) =================

run_pipeline() {
    local EXP_NAME=$1
    local MODEL_PATH=$2

    # 自动生成路径
    local STEP1_OUT="${BASE_RESULT_DIR}/${EXP_NAME}"
    local STEP2_OUT="${BASE_RESULT_DIR}/${EXP_NAME}-out"
    local LOG_DIR="${BASE_RESULT_DIR}/logs_${EXP_NAME}"
    
    mkdir -p "$BASE_RESULT_DIR"
    mkdir -p "$LOG_DIR"

    echo "############################################################"
    echo "🟢 [开始任务] ${EXP_NAME}"
    echo "📦 [模型路径] ${MODEL_PATH}"
    echo "🕒 [开始时间] $(date)"
    echo "############################################################"

    set -e # 遇到错误停止当前函数执行

    # --- Step 1: 并行推理 ---
    echo -e "\n▶️ [1/3] Step 1: 启动并行推理 ($NUM_WORKERS 进程)..."
    
    for ((i=0; i<NUM_WORKERS; i++)); do
        # ⚠️ 注意：这里用了您提供的文件名 a-infer_batch_Real-final-batch
        $PYTHON_CMD /home/szq/moge2/MoGe/moge/scripts/a-infer_batch_Real-final-batch.py \
            --model_path "$MODEL_PATH" \
            --input_root "$DATASET_INPUT" \
            --output_root "$STEP1_OUT" \
            --project_root "$PROJECT_ROOT" \
            --total_shards "$NUM_WORKERS" \
            --shard_id "$i" \
            > "${LOG_DIR}/worker_${i}.log" 2>&1 &
        
        # 记录 PID 以便后续追踪（可选）
        # pids[${i}]=$! 
    done

    # 等待当前任务的所有分片跑完
    wait 
    echo "✅ [1/3] Step 1 并行推理完成。"

    # --- Step 2: 提取 ---
    echo -e "\n▶️ [2/3] Step 2: 提取 NPY..."
    $PYTHON_CMD /home/szq/moge2/MoGe/moge/scripts/b-ex-infer-ex-Real.py \
        --input_dir "$STEP1_OUT" \
        --output_dir "$STEP2_OUT" \
        2>&1 | tee "${LOG_DIR}/step2.log"
    echo "✅ [2/3] Step 2 完成。"

    # --- Step 3: 评估 ---
    echo -e "\n▶️ [3/3] Step 3: 评估..."
    $PYTHON_CMD /home/szq/moge2/MoGe/moge/scripts/c-real-tasks-pitch-altitude-fov-syn-debug-batch-classes4.py \
        --pred_root "$STEP2_OUT" \
        --gt_root "$GT_ROOT" \
        --csv_path "$CSV_PATH" \
        2>&1 | tee "${LOG_DIR}/step3.log"
    echo "✅ [3/3] Step 3 完成。"

    echo -e "\n🎉 任务 ${EXP_NAME} 全部结束！\n"
    sleep 3 # 休息几秒，让 GPU 显存彻底释放，防止下一个任务瞬间 OOM
}

# ================= 🚀 启动引擎 =================

echo "🚀 批量排队系统启动..."
echo "总并行数 (Per Task): $NUM_WORKERS"

# 执行上面定义的任务列表
run_all_tasks

# echo "🏆🏆🏆 所有排队任务均已处理完毕！"