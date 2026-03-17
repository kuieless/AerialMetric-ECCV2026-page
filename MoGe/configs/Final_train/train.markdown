# 假设您在 MoGe 根目录下
# 记得把 --checkpoint 指向您下载的 v2 权重文件

CUDA_VISIBLE_DEVICES=3 accelerate launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    moge/scripts/train.py \
    --enable_ema False \
    --config /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/GAU-train/train.json \
    --workspace workspace/finetune_GAU_v2_run_6all \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 1 \
    --checkpoint /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/model.pt \
    --enable_gradient_checkpointing False \
    --vis_every 1000 \
    --save_every 400 \
    --enable_mlflow True



没问题，为了方便您直接复制粘贴，我将去掉所有复杂的 Markdown 格式（如加粗、代码块等），改为清晰的纯文本格式。您可以直接复制下面的内容，不会有乱码。

以下是关于 MoGe-2 配置文件中各个参数的详细解释：

================================================================ MoGe-2 配置文件详细解析 (纯文本版)
数据加载与增强 (data 部分) 这一部分决定了喂给模型的数据长什么样。

aspect_ratio_range (宽高比范围): [0.5, 2.0] 含义：训练时随机裁剪图片的形状范围，从 1:2 (高条形) 到 2:1 (扁条形)。

area_range (面积范围): [250000, 1000000] 含义：训练图片的像素总数范围。例如 25万像素对应约 500x500 分辨率，100万对应约 1000x1000。

clamp_max_depth (最大深度截断): 1000.0 含义：超过 1000 米的深度值会被强制截断。对于航拍，防止无穷远处的数值导致计算不稳定。

fov_range_absolute & fov_range_relative (视场角范围) 含义：几何增强参数。训练时会模拟相机焦距 (FOV) 的变化，这对让模型适应不同航拍镜头非常关键。

datasets (数据集列表)

label_type (标签类型 A/B/C): 这是最关键的参数。它决定了该数据集使用哪一种 Loss 组合。

A类: 高质量数据 (如合成数据)，计算所有 Loss (含法向、Mask)。

B类: 真实数据 (如 SfM 重建)，可能有噪声。

C类: 稀疏或低质量数据。

weight (权重): 决定该数据集被抽到的概率。

depth_unit (深度单位换算): 非常重要。如果您的原始数据单位是毫米，这里要填 0.001 把它换算成“米”。MoGe-2 必须统一单位才能学习真实尺度。

模型架构 (model 部分) 这一部分定义了神经网络的结构。

encoder (编码器)

backbone: 使用 "dinov2_vitl14" (ViT-Large)。

intermediate_layers: 提取第 6, 12, 18, 24 层特征用于融合。

neck (颈部网络)

dim_in: 输入通道配置。第一层是 1026 (1024特征 + 2坐标)，后续层只有 2 (坐标) 。

resamplers: 上采样方式，前几层用反卷积 (conv_transpose)，最后一层用双线性插值。

points_head (点云头)

输出 3 通道 (x, y, z) 的相对点云。

scale_head (尺度头)

这是一个 3 层的 MLP 网络。输入 1024 维的 CLS Token，输出 1 个标量 (全局尺度)。

remap_output (输出映射): "exp" 含义：模型预测的是“对数深度”。代码会执行 exp(z) 将其还原为正数，保证深度永远大于 0 。

num_tokens_range (Token 数量范围): [1200, 3600] 含义：训练时动态改变分辨率，增强模型对不同清晰度图像的适应能力 。


优化器与调度 (optimizer & lr_scheduler 部分)

差分学习率

头部网络 (Head/Neck) 使用较大的学习率 (1e-4)。

主干网络 (Backbone) 使用较小的学习率 (1e-5)，因为 DINOv2 已经预训练得很好了。

调度策略

前 1000 步进行预热 (Warmup)，学习率线性增加。

之后每 25000 步学习率减半 (StepLR)。

损失函数 (loss 部分) - 微调核心 这部分根据 label_type (A/B/C) 决定怎么算分。以下以 Type A 为例：

global (全局几何 Loss) 作用：先将点云缩放到 48x48 进行对齐，确保整体大轮廓正确 。

patch_4 / patch_16 / patch_64 (局部几何 Loss) 作用：在不同精细度上随机采样局部区域进行对齐。Patch 64 代表非常细小的局部，强迫模型学会精细纹理。

metric_scale (尺度 Loss) 作用：监督 Scale Head。计算预测尺度与真实尺度的对数均方误差 。 微调建议：如果您发现模型预测的航拍高度/尺寸不准，请调大这个权重 (目前是 0.1)。

normal (法向/边缘 Loss) 作用：比较相邻像素的差分，让物体表面更平滑 。

mask (掩膜 Loss) 作用：学习区分天空、背景和前景物体 。

================================================================ 针对航拍任务的微调建议
调整 fov_range 航拍相机通常焦距固定或为广角，请根据您实际数据的 FOV 分布调整 fov_range_absolute，不要使用默认的 1-179 这么大的范围。

调整 Loss 权重

增加 metric_scale 的权重：航拍图覆盖范围大，尺度极难预测。建议将 metric_scale 的权重从 0.1 提高到 0.5 或 1.0，强迫模型关注绝对高度/尺寸。

增加 patch_64 的权重：如果航拍图中的小物体 (车、树) 模糊，增加这个权重有助于恢复高频细节。

调整 clamp_max_depth 如果是高空航拍 (如 500米以上)，默认的 1000.0 可能不够，建议根据数据集的最大高度进行调整。

设置 Label Type

如果您有精准的 LiDAR 真值，请将数据集设为 Type A。

如果是 SfM 重建数据 (可能有噪声)，设为 Type B。











================================================================ 针对航拍任务的微调建议
# 指定使用 4 张卡 (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3,4 accelerate launch \
    --num_processes 5 \
    --mixed_precision bf16 \
    moge/scripts/train.py \
    --enable_ema False \
    --config /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/GAU-train/train.json \
    --workspace workspace/Infer6-all-finedata-check-l \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 1 \
    --checkpoint /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing False \
    --vis_every 2000 \
    --save_every 1000 \
    --enable_mlflow True




CUDA_VISIBLE_DEVICES=2 accelerate launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    moge/scripts/train.py \
    --enable_ema False \
    --config /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/GAU-train/train.json \
    --workspace workspace/Infer6-all-finedata-check-l2 \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 1 \
    --checkpoint /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vits-normal.pt \
    --enable_gradient_checkpointing False \
    --vis_every 1000 \
    --save_every 1000 \
    --enable_mlflow True


CUDA_VISIBLE_DEVICES=2 accelerate launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    moge/scripts/train.py \
    --enable_ema False \
    --config /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/GAU-train/train-s-freeze.json \
    --workspace workspace/Infer6-all-finedata-check-s-freeze \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 1 \
    --checkpoint /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vits-normal.pt \
    --enable_gradient_checkpointing False \
    --vis_every 1000 \
    --save_every 1000 \
    --enable_mlflow True

CUDA_VISIBLE_DEVICES=0,1,2,3,4 accelerate launch \
    --num_processes 5 \
    --mixed_precision bf16 \
    moge/scripts/train.py \
    --enable_ema True \
    --config /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/GAU-train/train-l-scalehead.json \
    --workspace workspace/Infer6-large-max-throughput2-12.12-vitl-1-loss2.0-scalehead \
    --gradient_accumulation_steps 8 \
    --batch_size_forward 1 \
    --checkpoint /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing False \
    --vis_every 500 \
    --save_every 500 \
    --enable_mlflow True




        //     "gradient": { 
        //     "function": "multi_scale_gradient_loss", 
        //     "weight": 1.5, 
        //     "params": {"scales": 4} 
        // }


    命令参数详解（必读）：
--num_processes 4: 明确告诉 accelerate 使用 4 个进程（对应 4 张卡）。

--mixed_precision fp16 (accelerate 参数) & --enable_mixed_precision True (train.py 参数): 双重确认开启混合精度。3090 必须开这个，否则显存不够，且速度慢。

--batch_size_forward 1: 每张卡一次只处理 1 张图。这是为了防止 24G 显存溢出。

--gradient_accumulation_steps 8: 因为单次前向传播总共只有 4 张图 (1*4)，我们让模型攒够 8 次（即处理了 32 张图）后再更新一次参数。这样等效 Batch Size 就是 32，保证了训练的稳定性。

--enable_gradient_checkpointing True: 必须开启。这是以时间换空间的策略，能大幅降低显存占用，支持 ViT-Large 在 3090 上运行。

--enable_ema False: 微调时通常不需要 EMA (指数移动平均) 模型，除非你要训练很久。关掉它可以节省显存和磁盘空间。如果您确实需要 EMA 模型，可以改回 True。

运行前检查： 请确保 /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/GAU-train/train.json 这个文件里的内容确实是您上面发给我的那一段（特别是 datasets 路径要对）。


/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/

CUDA_VISIBLE_DEVICES=2,3,4,5,6 accelerate launch \
    --num_processes 5 \
    --mixed_precision bf16 \
    moge/scripts/trainall.py \
    --enable_ema True \
    --config /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/Final_train/train-l.json \
    --workspace workspace/final-fintune2-1.18 \
    --gradient_accumulation_steps 8 \
    --batch_size_forward 2 \
    --checkpoint /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 500 \
    --save_every 500 \
    --enable_mlflow False


CUDA_VISIBLE_DEVICES=1,5,6 accelerate launch \
    --num_processes 3 \
    --mixed_precision bf16 \
    moge/scripts/trainall.py \
    --enable_ema True \
    --config /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/Final_train/train-l-patch.json \
    --workspace workspace/final-fintune2-1.18-multiloss-patch8-normal3 \
    --gradient_accumulation_steps 2 \
    --batch_size_forward 8 \
    --checkpoint /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 500 \
    --save_every 500 \
    --enable_mlflow False






CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch \
    --num_processes 3 \
    --mixed_precision bf16 \
    moge/scripts/trainall-head-neck.py \
    --enable_ema True \
    --config /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/Final_train/train-l-scalehead-neck.json \
    --workspace workspace/final-fintune-scalehead-neck \
    --gradient_accumulation_steps 8 \
    --batch_size_forward 2 \
    --checkpoint /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 500 \
    --save_every 500 \
    --enable_mlflow True



CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch \
    --num_processes 3 \
    --mixed_precision bf16 \
    moge/scripts/trainall-head-neck.py \
    --enable_ema True \
    --config /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/Final_train/train-l-scalehead-neck.json \
    --workspace workspace/final-fintune-scalehead-neck \
    --gradient_accumulation_steps 8 \
    --batch_size_forward 2 \
    --checkpoint /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 500 \
    --save_every 500 \
    --enable_mlflow True

CUDA_VISIBLE_DEVICES=7 ./all.sh



CUDA_VISIBLE_DEVICES=7 accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    moge/scripts/trainall-head-neck.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/train-l-scalehead-neck.json \
    --workspace workspace/final-fintune-scalehead-neck119 \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 8 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing False \
    --vis_every 500 \
    --save_every 500 \
    --enable_mlflow True



CUDA_VISIBLE_DEVICES=7 ./all.sh


1-allpatch

CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch \
    --num_processes 3 \
    --mixed_precision bf16 \
    moge/scripts/trainall-head-neck.py \
    --enable_ema True \
    --config /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/configs/Final_train/train-l-scalehead-neck.json \
    --workspace workspace/final-fintune-scalehead-neck \
    --gradient_accumulation_steps 8 \
    --batch_size_forward 2 \
    --checkpoint /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 500 \
    --save_every 500 \
    --enable_mlflow True



lora


CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/train-l-patch-lora.json \
    --workspace workspace/final-fintune-scalehead-lora1 \
    --gradient_accumulation_steps 2 \
    --batch_size_forward 8 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow True \
    --enable_mixed_precision False


CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/train-l-patch-lora.json \
    --workspace workspace/final-fintune-scalehead-lora2 \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 2 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 500 \
    --save_every 500 \
    --enable_mlflow True \
    --enable_mixed_precision False























<!-- ============================================================================= -->
# 1. 创建 data 目录（如果不存在）
mkdir -p data

# 2. 建立强行链接 (注意路径不要写错)
ln -sfn /home/data1/szq/Megadepth/Benchmark-final2/moge-eva data/eval

python moge/scripts/eval_baseline.py \
  --baseline baselines/moge.py \
  --config configs/eval/all_benchmarks.json \
  --output eval_output/my_vitl_normal_eval.json \
  --pretrained /home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vitl-normal.pt \
  --resolution_level 9 \
  --dump_pred

  /home/szq/moge2




  HEAD：
  CUDA_VISIBLE_DEVICES=7 accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/train-l-patch.json \
    --workspace workspace/final-fintune2-1.18-multiloss-patch8-normal4-122-5-lanzocos \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 1 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow False

  CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/train-l-patch-lora.json \
    --workspace workspace/final-fintune2-130-lora-final6-64r \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 500 \
    --save_every 500 \
    --enable_mlflow False


 CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/train-l-patch-lora.json \
    --workspace workspace/final-fintune2-127-lora1-r8 \
    --batch_size_forward 1 \
    --base_checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --num_vis_images 8 


# 假设你使用 6号 和 7号 显卡 (根据实际情况修改)
CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --num_processes 2 \
    --main_process_port 29500 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/train-l-patch-lora.json \
    --workspace workspace/final-fintune2-127-lora1-r8 \
    --base_checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --batch_size_forward 4 \
    --gradient_accumulation_steps 4 \
    --save_every 500 \
    --vis_every 500 \
    --num_vis_images 8 \
    --seed 42

accelerate launch --multi_gpu --num_processes 8 train_lora.py \
  --config configs/train_lora.json \
  --workspace workspace/lora_run_01 \
  --base_checkpoint /path/to/your/pretrained_moge_v2.pt \
  --batch_size_forward 4 \
  --vis_every 1000 \
  --num_vis_images 8






CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    moge/scripts/trainall-head.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/train-l-patch-head.json \
    --workspace workspace/final-fintune-scalehead-head124 \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 8 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing False \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow True


CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    moge/scripts/trainall-neck.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/train-l-patch-neck.json \
    --workspace workspace/final-fintune-scalehead-neck123-origin-loss3 \
    --gradient_accumulation_steps 1 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 500 \
    --save_every 500 \
    --enable_mlflow True




=============================================================

 
  CUDA_VISIBLE_DEVICES=5,6,7 accelerate launch \
    --num_processes 3 \
    --mixed_precision bf16 \
    moge/scripts/trainall.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config.json \
    --workspace workspace/final-ground \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 300 \
    --save_every 300 \
    --enable_mlflow False










Neck：
  CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-neck.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-neck.json \
    --workspace /data1/szq/workspace/final-neck-lossfine \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 600 \
    --save_every 600 \
    --enable_mlflow False \
    --seed 333

Head：
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-head.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-head.json \
    --workspace /data1/szq/workspace/final-head \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 600 \
    --save_every 600 \
    --enable_mlflow False \
    --seed 42



全参数：
  CUDA_VISIBLE_DEVICES=0,3 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-all.json \
    --workspace /data1/szq/workspace/final-all \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 600 \
    --save_every 600 \
    --enable_mlflow False \
    --seed 333





lora：跑多久呢？
1、改/home/szq/moge2/MoGe/moge/scripts/trainall-lora.py 里的rank和alpha的值
2、

Lora:64-128
  CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora64-128.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-nosyn.json \
    --workspace workspace/lora-batch64-128 \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow False \
    --seed 42




Lora:32-64
  CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora32-64.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UE.json \
    --workspace /data1/szq/workspace/lora-batch32-64-lessground4-with-UE \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow False \
    --seed 42



Lora:8-16
  CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora8-16.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UE.json \
    --workspace workspace/lora-batch8-16-lessground4-with-UE \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow False \
    --seed 333


  CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora8-16.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-no.json \
    --workspace workspace/lora-batch8-16-lessground4-onlyaerial \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow False \
    --seed 333
    

Lora:16-32
  CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora16-32.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all.json \
    --workspace workspace/lora-batch16-32-lessground4-with-UE \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow False \
    --seed 333

Lora:64-128
  CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora64-128.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UElr2.json \
    --workspace /data1/szq/workspace/lora-batch64-128-with-UElr2 \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow False \
    --seed 42

Lora:64-128
  CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora96-192.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UElr2.json \
    --workspace /data1/szq/workspace/lora-batch96-192-with-UElr2 \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow False \
    --seed 42

  CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora96-192.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UElr2.json \
    --workspace /data1/szq/workspace/lora-batch96-192-with-UElr2 \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow False \
    --seed 42


  CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora96-192.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UElr2-aerial.json \
    --workspace /data1/szq/workspace/lora-batch96-192-with-UElr2-aerial4 \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 100 \
    --save_every 100 \
    --enable_mlflow False \
    --seed 333

  CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora96-192.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UElr2-aerial-UE.json \
    --workspace /data1/szq/workspace/lora-batch96-192-with-UElr2-aerial-UE \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 100 \
    --save_every 100 \
    --enable_mlflow False \
    --seed 333



Lora:128-256
  CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora128-256.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UElr2.json \
    --workspace /data1/szq/workspace/lora-batch128-256-lessground4-with-UElr2-3 \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 100 \
    --save_every 100 \
    --enable_mlflow False \
    --seed 333


Lora:256-512
  CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    moge/scripts/trainall-lora256-512.py \
    --enable_ema True \
    --config /home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UElr2.json \
    --workspace /data1/szq/workspace/lora-batch256-512-lessground4-with-UElr2 \
    --gradient_accumulation_steps 4 \
    --batch_size_forward 4 \
    --checkpoint /home/szq/moge2/MoGe/vitl-normal.pt \
    --enable_gradient_checkpointing True \
    --vis_every 50 \
    --save_every 50 \
    --enable_mlflow False \
    --seed 333

/home/szq/moge2/MoGe/configs/Final_train/config-lora-all-UElr2.json