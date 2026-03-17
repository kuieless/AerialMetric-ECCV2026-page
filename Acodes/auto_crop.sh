# #!/bin/bash

# # ================= 核心配置区域 =================

# # 1. Python 脚本的绝对路径 (建议写绝对路径，防止找不到)
# PY_SCRIPT="/path/to/your/auto_crop.py"

# # 2. 裁剪倍数
# DIVISOR=14

# # 3. 任务列表
# # 格式： "输入绝对路径|输出绝对路径"
# # 注意：中间用竖线 | 隔开，不要有空格（除非路径本身包含空格）
# TASKS=(
#     "/home/user/data/dataset_A/images|/home/user/data/dataset_A/cropped"
#     "/mnt/disk2/aerial_photos/scene_01|/mnt/disk2/aerial_photos/scene_01_v2"
#     "/data/raw/test_set|/data/processed/test_set_crop"
# )

# # ================= 执行逻辑 (无需修改) =================

# echo "==========================================="
# echo "Batch Adaptive Crop (Absolute Paths)"
# echo "Target Divisor: $DIVISOR"
# echo "Task Count: ${#TASKS[@]}"
# echo "==========================================="

# # 遍历任务列表
# for entry in "${TASKS[@]}"; do
    
#     # 使用 IFS (Internal Field Separator) 将字符串按 | 分割
#     # input_dir 拿到前半部分， output_dir 拿到后半部分
#     IFS='|' read -r input_dir output_dir <<< "$entry"
    
#     # 去除可能存在的首尾空白字符 (以防万一)
#     input_dir=$(echo "$input_dir" | xargs)
#     output_dir=$(echo "$output_dir" | xargs)

#     echo ""
#     echo ">>> Starting Task..."
#     echo "    Input:  $input_dir"
#     echo "    Output: $output_dir"

#     # 1. 检查输入目录是否存在
#     if [ ! -d "$input_dir" ]; then
#         echo "❌ ERROR: Input directory does not exist!"
#         echo "   Path: $input_dir"
#         echo "   Skipping this task..."
#         continue
#     fi

#     # 2. 执行 Python 脚本
#     # 这里的 python 建议使用具体的环境路径，例如 /home/user/anaconda3/envs/myenv/bin/python
#     python "$PY_SCRIPT" \
#         --input "$input_dir" \
#         --output "$output_dir" \
#         --divisor "$DIVISOR"

#     # 3. 检查执行结果
#     if [ $? -eq 0 ]; then
#         echo "✅ Task Finished."
#     else
#         echo "❌ Task Failed."
#     fi

# done

# echo ""
# echo "==========================================="
# echo "🎉 All Tasks Processed."
# echo "==========================================="


#!/bin/bash

# ================= 核心配置区域 =================

# 1. Python 脚本路径
PY_SCRIPT="/home/data1/szq/Megadepth/benchemarkdata/Acodes/auto_crop.py"

# 2. 裁剪倍数
DIVISOR=14

# 3. 任务列表
# 格式： "图片输入|图片输出|NPY输入|NPY输出"
# 如果不需要处理 NPY，后两项可以留空或者写 "None" (但建议 Python 脚本里处理 None)
# 这里假设一定要处理 NPY
TASKS=(
    # "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/images_gt_hav/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GAU/images_gt_hav|/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_hav|/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_hav"
    # "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/images_gt_lfls/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GAU/images_gt_lfls|/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_lfls|/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_lfls"
    #     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/images_gt_lfls2/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GAU/images_gt_lfls2|/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_lfls2|/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_lfls2"
    #         "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/images_gt_lower/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GAU/images_gt_lower|/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_lower|/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_lower"
    #             "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/images_gt_SMBU/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GAU/images_gt_SMBU|/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_SMBU|/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_SMBU"
    #                 "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/images_gt_sziit/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GAU/images_gt_sziit|/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_sziit|/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_sziit"
    #                     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/images_gt_sztu/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GAU/images_gt_sztu|/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_sztu|/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_sztu"
    #                         "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/images_gt_upper/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GAU/images_gt_upper|/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/depth_gt_upper|/home/data1/szq/Megadepth/benchemarkdata/GAU/depth_gt_upper"


        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/dj/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/dj/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj/npy"  
        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/dj2/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj2/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/dj2/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj2/npy"
        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/dj3/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj3/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/dj3/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj3/npy"
        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/dj4/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj4/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/dj4/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/dj4/npy"
        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/hsd1/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/hsd1/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/hsd1/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/hsd1/npy"
        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/lm/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/lm/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/lm/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/lm/npy"
        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/xg1/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg1/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/xg1/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg1/npy"
        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/xg2/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg2/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/xg2/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg2/npy"
        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/xg3/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg3/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/xg3/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg3/npy"
        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/xg4/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg4/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/xg4/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg4/npy"
        # "/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/xg5/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg5/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin/xg5/npy|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-origin-crop/xg5/npy"


"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/dj/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/dj/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj/npy"  
"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/dj2/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj2/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/dj2/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj2/npy"
"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/dj3/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj3/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/dj3/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj3/npy"
"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/dj4/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj4/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/dj4/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/dj4/npy"
"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/hsd1/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/hsd1/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/hsd1/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/hsd1/npy"
"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/lm/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/lm/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/lm/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/lm/npy"
"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/xg1/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg1/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/xg1/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg1/npy"
"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/xg2/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg2/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/xg2/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg2/npy"
"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/xg3/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg3/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/xg3/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg3/npy"
"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/xg4/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg4/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/xg4/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg4/npy"
"/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/xg5/images_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg5/images|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down/xg5/npy_downsampled|/home/data1/szq/Megadepth/benchemarkdata/GES11/GT-down-crop/xg5/npy"






)
# ================= 执行逻辑 =================

echo "==========================================="
echo "Batch Adaptive Crop (Image + NPY Sync)"
echo "Target Divisor: $DIVISOR"
echo "==========================================="

for entry in "${TASKS[@]}"; do
    
    # 使用 IFS 将字符串分割为 4 个变量
    IFS='|' read -r img_in img_out npy_in npy_out <<< "$entry"
    
    # 去除可能的空格
    img_in=$(echo "$img_in" | xargs)
    img_out=$(echo "$img_out" | xargs)
    npy_in=$(echo "$npy_in" | xargs)
    npy_out=$(echo "$npy_out" | xargs)

    echo ""
    echo ">>> Starting Task..."
    echo "    Img In:  $img_in"
    echo "    Img Out: $img_out"
    echo "    Npy In:  $npy_in"
    echo "    Npy Out: $npy_out"

    # 基本检查
    if [ ! -d "$img_in" ]; then
        echo "❌ ERROR: Image directory missing: $img_in"
        continue
    fi
    
    if [ ! -d "$npy_in" ]; then
        echo "❌ ERROR: NPY directory missing: $npy_in"
        continue
    fi

    # 执行 Python 脚本
    python "$PY_SCRIPT" \
        --input "$img_in" \
        --output "$img_out" \
        --input_npy "$npy_in" \
        --output_npy "$npy_out" \
        --divisor "$DIVISOR"

    if [ $? -eq 0 ]; then
        echo "✅ Task Finished."
    else
        echo "❌ Task Failed."
    fi

done

echo ""
echo "==========================================="
echo "🎉 All Tasks Processed."