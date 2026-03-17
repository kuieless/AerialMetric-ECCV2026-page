import numpy as np
import os
from glob import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import warnings

# 忽略一些除以0的警告
warnings.filterwarnings("ignore")

# =================================================================================
# 1. 路径列表定义区域
# =================================================================================





# PATHS_DAV2_LARGE = [
#     {
#         'name': 'lower',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-l/lower',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/lower-npy'
#     },
#     {
#         'name': 'sziit',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-l/sziit',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/sziit-npy'
#     },
#     {
#         'name': 'dj3',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-l/dj3',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/dj3-npy'
#     },
#     {
#         'name': 'hsd1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-l/hsd1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/hsd1-npy'
#     },
#     {
#         'name': 'xg5',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-l/xg5',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/xg5-npy'
#     },
#     {
#         'name': 'town1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-l/town1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town1-npy'
#     },
#     {
#         'name': 'town2',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-l/town2',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town2-npy'
#     },
#     {
#         'name': 'town3',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-l/town3',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town3-npy'
#     },
#     {
#         'name': 'yingrenshi1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-l/yingrenshi1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/yingrenshi1-npy'
#     },
#     {
#         'name': 'yingrenshi2',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-l/yingrenshi2',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/yingrenshi2-npy'
#     },
#     {
#         'name': 'SYS',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-l/SYS',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/SYS-npy'
#     },
#     {
#         'name': 'yuehai',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-l/yuehai',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/yuehai-npy'
#     },
# ]

# PATHS_DAV2_SMALL = [
#     {
#         'name': 'lower',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-s/lower',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/lower-npy'
#     },
#     {
#         'name': 'sziit',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-s/sziit',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/sziit-npy'
#     },
#     {
#         'name': 'dj3',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-s/dj3',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/dj3-npy'
#     },
#     {
#         'name': 'hsd1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-s/hsd1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/hsd1-npy'
#     },
#     {
#         'name': 'xg5',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-s/xg5',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/xg5-npy'
#     },
#     {
#         'name': 'town1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-s/town1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town1-npy'
#     },
#     {
#         'name': 'town2',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-s/town2',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town2-npy'
#     },
#     {
#         'name': 'town3',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-s/town3',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town3-npy'
#     },
#     {
#         'name': 'yingrenshi1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-s/yingrenshi1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/yingrenshi1-npy'
#     },
#     {
#         'name': 'yingrenshi2',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-s/yingrenshi2',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/yingrenshi2-npy'
#     },
#     {
#         'name': 'SYS',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-s/SYS',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/SYS-npy'
#     },
#     {
#         'name': 'yuehai',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-s/yuehai',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/yuehai-npy'
#     },
# ]


# PATHS_DAV2_BASE = [
#     {
#         'name': 'lower',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-b/lower',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/lower-npy'
#     },
#     {
#         'name': 'sziit',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-b/sziit',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/sziit-npy'
#     },
#     {
#         'name': 'dj3',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-b/dj3',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/dj3-npy'
#     },
#     {
#         'name': 'hsd1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-b/hsd1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/hsd1-npy'
#     },
#     {
#         'name': 'xg5',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-b/xg5',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/xg5-npy'
#     },
#     {
#         'name': 'town1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-b/town1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town1-npy'
#     },
#     {
#         'name': 'town2',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-b/town2',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town2-npy'
#     },
#     {
#         'name': 'town3',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-b/town3',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/town3-npy'
#     },
#     {
#         'name': 'yingrenshi1',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-b/yingrenshi1',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/yingrenshi1-npy'
#     },
#     {
#         'name': 'yingrenshi2',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer1/Infer-b/yingrenshi2',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/yingrenshi2-npy'
#     },
#     {
#         'name': 'SYS',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-b/SYS',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/SYS-npy'
#     },
#     {
#         'name': 'yuehai',
#         'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer2/Infer-b/yuehai',
#         'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/yuehai-npy'
#     },
# ]


PATHS_DAV2_LARGE = [

        {
        'name': 'lower',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/lower',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/lower-npy'
    },
    {
        'name': 'sziit',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/sziit',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/sziit-npy'
    },
    {
        'name': 'dj3',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/dj3',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/dj3-npy'
    },
    {
        'name': 'hsd1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/hsd1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/hsd1-npy'
    },
    {
        'name': 'xg5',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/xg5',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/xg5-npy'
    },
    {
        'name': 'town1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/town1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town1-npy'
    },
    {
        'name': 'town2',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/town2',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town2-npy'
    },
    {
        'name': 'town3',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/town3',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/town3-npy'
    },
    {
        'name': 'yingrenshi1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/yingrenshi1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/yingrenshi1-npy'
    },
    {
        'name': 'yingrenshi2',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/yingrenshi2',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test/yingrenshi2-npy'
    },
    {
        'name': 'SYS',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/SYS',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test2/SYS-npy'
    },
    {
        'name': 'yuehai',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/yuehai',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/yuehai-npy'
    },
    {
        'name': 'D1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/D1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/D1-npy'
    },
    {
        'name': 'R1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/R1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/R1-npy'
    },
    {
        'name': 'R1-PHD',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/R1-PHD',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/R1-PHD-npy'
    },
    {
        'name': 'bellus',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/bellus',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/bellus-npy'
    },
    {
        'name': 'brighton-beach',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/brighton-beach',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/brighton-beach-npy'
    },
    {
        'name': 'ODM1',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/ODM1',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/ODM1-npy'
    },
    {
        'name': 'park13',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/park13',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/park13-npy'
    },
    {
        'name': 'seneca',
        'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer/AAA-down-all/seneca',
        'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-down-all/seneca-npy'
    },


# {
#     'name': 'D1',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer3/Infer-l/D1',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/D1-npy'
# },
# {
#     'name': 'R1',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer3/Infer-l/R1',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/R1-npy'
# },
# {
#     'name': 'R1-PHD',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer3/Infer-l/R1-PHD',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/R1-PHD-npy'
# },
 
]

# PATHS_DAV2_SMALL = [
# {
#     'name': 'D1',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer3/Infer-s/D1',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/D1-npy'
# },
# {
#     'name': 'R1',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer3/Infer-s/R1',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/R1-npy'
# },
# {
#     'name': 'R1-PHD',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer3/Infer-s/R1-PHD',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/R1-PHD-npy'
# },
# ]


PATHS_DAV2_BASE = [

#   {
#     'name': 'D1',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer3/Infer-b/D1',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/D1-npy'
# },
# {
#     'name': 'R1',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer3/Infer-b/R1',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/R1-npy'
# },
# {
#     'name': 'R1-PHD',
#     'pred_dir': '/home/data1/szq/Megadepth/benchmarkmodel/Depthanythingv2/infer3/Infer-b/R1-PHD',
#     'gt_dir': '/home/data1/szq/Megadepth/benchemarkdata/AAA-test3/R1-PHD-npy'
# },  
]


# =================================================================================
# 2. 任务配置区域
# =================================================================================

TASKS = [
      {
        "task_name": "DAV2-LARGE-ScaleShift",
        "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/DAV2/Infer-AAA-down-l.txt",
        "use_relative_disparity_mode": True, 
        "paths": PATHS_DAV2_LARGE
    },
    #   {
    #     "task_name": "DAV2-SMALL-ScaleShift",
    #     "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/DAV2/Infer3-ScaleShift-s_Binned.txt",
    #     "use_relative_disparity_mode": True, 
    #     "paths": PATHS_DAV2_SMALL
    # },
    #       {
    #     "task_name": "DAV2-BASE-ScaleShift",
    #     "output_file": "/home/data1/szq/Megadepth/benchemarkdata/AAA-Infertxt/DAV2/Infer3-ScaleShift-g_Binned.txt",
    #     "use_relative_disparity_mode": True, 
    #     "paths": PATHS_DAV2_BASE
    # },
]

# =================================================================================
# 3. 核心配置: 深度分段 (与第一个脚本保持一致)
# =================================================================================
DEPTH_BINS = [
    (0, 50),
    (50, 120),
    (120, 250),
    (250, 400),
    (400, 500)
]
BIN_LABELS = [f"{b[0]}-{b[1]}m" for b in DEPTH_BINS]
ALL_LABELS = BIN_LABELS + ["Overall"]

BATCH_SIZE = 16  # Scale&Shift 矩阵运算比较吃显存
NUM_WORKERS = 8
MIN_EVAL_DEPTH = 1e-3
MAX_EVAL_DEPTH = 500

# =================================================================================
# 数据加载器
# =================================================================================

class PairedNpyDataset(Dataset):
    def __init__(self, pred_dir, gt_dir):
        all_pred_files = sorted(glob(os.path.join(pred_dir, "*.npy")))
        self.pred_files = []
        self.gt_files = []
        
        for pred_path in all_pred_files:
            basename = os.path.basename(pred_path)
            gt_path = os.path.join(gt_dir, basename)
            if os.path.exists(gt_path):
                self.pred_files.append(pred_path)
                self.gt_files.append(gt_path)
        
        # 递归查找备选方案
        if len(self.pred_files) == 0:
            all_pred_files = sorted(glob(os.path.join(pred_dir, "**/*.npy"), recursive=True))
            for pred_path in all_pred_files:
                basename = os.path.basename(pred_path)
                possible_gts = glob(os.path.join(gt_dir, "**", basename), recursive=True)
                if len(possible_gts) > 0:
                     self.pred_files.append(pred_path)
                     self.gt_files.append(possible_gts[0])

        if len(self.pred_files) == 0:
             print(f"Warning: No matched files in {pred_dir}")

    def __len__(self): return len(self.pred_files)

    def __getitem__(self, idx):
        try:
            pred_np = np.load(self.pred_files[idx]).astype(np.float32)
            gt_np = np.load(self.gt_files[idx]).astype(np.float32)
            if pred_np.ndim == 3: pred_np = np.squeeze(pred_np)
            if gt_np.ndim == 3: gt_np = np.squeeze(gt_np)
            return torch.from_numpy(pred_np), torch.from_numpy(gt_np)
        except Exception as e:
            print(f"Error loading {self.pred_files[idx]}: {e}")
            return torch.zeros((1,1)), torch.zeros((1,1))

# =================================================================================
# 4. 误差计算 (支持 Bins)
# =================================================================================

def compute_errors_torch_bins(gt, pred, valid_mask):
    """
    输入:
        gt: [B, H, W]
        pred: [B, H, W] (已经 Scale & Shift 对齐好的深度)
        valid_mask: [B, H, W] (主要掩码)
    输出:
        [Batch, Num_Bins + 1, 8]
    """
    # 1. 预处理
    gt_c = torch.clamp(gt, min=MIN_EVAL_DEPTH)
    pred_c = torch.clamp(pred, min=MIN_EVAL_DEPTH)
    
    # 2. 预计算 Error Maps
    rmse_map = (gt - pred) ** 2
    rmse_log_map = (torch.log(gt_c) - torch.log(pred_c)) ** 2
    abs_rel_map = torch.abs(gt - pred) / gt_c
    sq_rel_map = ((gt - pred) ** 2) / gt_c
    thresh_val = torch.maximum((gt_c / pred_c), (pred_c / gt_c))
    
    # 3. 定义区间：分段 + Overall
    all_ranges = DEPTH_BINS + [("Overall", "Overall")] 
    
    batch_results = []
    
    for (b_min, b_max) in all_ranges:
        if b_min == "Overall":
            current_mask = valid_mask
        else:
            # 这里的区间判定基于 GT 深度
            current_mask = valid_mask & (gt >= b_min) & (gt < b_max)
            
        mask_f = current_mask.float()
        valid_pixel_count = mask_f.sum(dim=[1, 2])
        
        # 统计各项指标
        a1 = ((thresh_val < 1.25) & current_mask).sum(dim=[1, 2]).float()
        a2 = ((thresh_val < 1.25 ** 2) & current_mask).sum(dim=[1, 2]).float()
        a3 = ((thresh_val < 1.25 ** 3) & current_mask).sum(dim=[1, 2]).float()
        
        rmse_s = (rmse_map * mask_f).sum(dim=[1, 2])
        rmse_log_s = (rmse_log_map * mask_f).sum(dim=[1, 2])
        abs_rel_s = (abs_rel_map * mask_f).sum(dim=[1, 2])
        sq_rel_s = (sq_rel_map * mask_f).sum(dim=[1, 2])
        
        # [Batch, 8]
        bin_res = torch.stack([abs_rel_s, sq_rel_s, rmse_s, rmse_log_s, a1, a2, a3, valid_pixel_count], dim=1)
        batch_results.append(bin_res)
        
    # Stack -> [Batch, Num_Bins+1, 8]
    return torch.stack(batch_results, dim=1)

def compute_metrics_from_sums(sums):
    total_valid = sums[7]
    if total_valid <= 0: return np.zeros(7)
    return np.array([
        sums[0]/total_valid, sums[1]/total_valid, np.sqrt(sums[2]/total_valid),
        sums[3]/total_valid, sums[4]/total_valid, sums[5]/total_valid, sums[6]/total_valid
    ])

# =================================================================================
# 5. Scale & Shift 对齐逻辑
# =================================================================================

def batch_least_squares_alignment(pred_inv, gt, mask):
    """
    solve: s * pred_inv + t = gt
    使用整个 Valid Mask 进行全局对齐
    """
    B = pred_inv.shape[0]
    aligned_pred = torch.zeros_like(pred_inv)
    
    for i in range(B):
        valid_mask = mask[i]
        if valid_mask.sum() < 10: 
            aligned_pred[i] = pred_inv[i] 
            continue
            
        y = gt[i][valid_mask]       # (N,)
        x = pred_inv[i][valid_mask] # (N,)
        
        ones = torch.ones_like(x)
        A = torch.stack([x, ones], dim=1)
        
        solution = torch.linalg.lstsq(A, y).solution
        s, t = solution[0], solution[1]
        
        aligned_pred[i] = pred_inv[i] * s + t
        
    return aligned_pred

# def evaluate_single_scene(pred_dir, gt_dir, scene_name, is_relative_mode):
#     device = torch.device("cuda")
    
#     dataset = PairedNpyDataset(pred_dir, gt_dir)
#     if len(dataset) == 0: return None
    
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
#     all_errs_scene = [] # Store [Batch, Bins, 8]
    
#     for pred_batch, gt_batch in tqdm(dataloader, desc=f"  -> {scene_name}", leave=False):
#         pred_batch, gt_batch = pred_batch.to(device), gt_batch.to(device)
        
#         # 1. 尺寸对齐
#         if pred_batch.shape[2:] != gt_batch.shape[2:]:
#             pred_batch = F.interpolate(pred_batch.unsqueeze(1), size=gt_batch.shape[2:], mode='bilinear', align_corners=False).squeeze(1)

#         # 2. 生成基础掩码
#         mask_batch = (gt_batch > MIN_EVAL_DEPTH) & (gt_batch < MAX_EVAL_DEPTH) & torch.isfinite(gt_batch)

#         # 3. 对齐处理 (Alignment)
#         # 注意：这里我们做全局对齐，而不是分段对齐。分段评估是在全局对齐后的结果上进行的。
#         if is_relative_mode:
#             # 相对视差模式 (DepthAnything): 1/x -> Scale & Shift
#             pred_process = 1.0 / (pred_batch + 1e-6)
#             pred_aligned = batch_least_squares_alignment(pred_process, gt_batch, mask_batch)
#         else:
#             # 绝对深度模式 (Metric3D等): Scale & Shift
#             pred_aligned = batch_least_squares_alignment(pred_batch, gt_batch, mask_batch)
            
#         pred_aligned.clamp_(min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)

#         # 4. 计算多区间误差 (Bin Evaluation)
#         # 传入对齐后的 pred 和 原始 gt
#         errors_batch = compute_errors_torch_bins(gt_batch, pred_aligned, mask_batch)
#         all_errs_scene.append(errors_batch.cpu().numpy())
            
#     if not all_errs_scene: return None
#     # [Total_Imgs, Num_Bins+1, 8]
#     return np.concatenate(all_errs_scene, axis=0)

def evaluate_single_scene(pred_dir, gt_dir, scene_name, is_relative_mode):
    device = torch.device("cuda")
    
    dataset = PairedNpyDataset(pred_dir, gt_dir)
    if len(dataset) == 0: return None
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    all_errs_scene = [] # Store [Batch, Bins, 8]
    
    for pred_batch, gt_batch in tqdm(dataloader, desc=f"  -> {scene_name}", leave=False):
        pred_batch, gt_batch = pred_batch.to(device), gt_batch.to(device)
        
        # ==================== 修改开始 ====================
        # 错误原代码: if pred_batch.shape[2:] != gt_batch.shape[2:]:
        # 错误原代码:     pred_batch = F.interpolate(..., size=gt_batch.shape[2:], ...)
        
        # 修正后: 使用 [-2:] 来获取 (Height, Width)
        if pred_batch.shape[-2:] != gt_batch.shape[-2:]:
            pred_batch = F.interpolate(
                pred_batch.unsqueeze(1), 
                size=gt_batch.shape[-2:],  # <--- 关键修改：取最后两个维度 (H, W)
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        # ==================== 修改结束 ====================

        # 2. 生成基础掩码
        mask_batch = (gt_batch > MIN_EVAL_DEPTH) & (gt_batch < MAX_EVAL_DEPTH) & torch.isfinite(gt_batch)

        # 3. 对齐处理 (Alignment)
        if is_relative_mode:
            # 相对视差模式 (DepthAnything): 1/x -> Scale & Shift
            pred_process = 1.0 / (pred_batch + 1e-6)
            pred_aligned = batch_least_squares_alignment(pred_process, gt_batch, mask_batch)
        else:
            # 绝对深度模式 (Metric3D等): Scale & Shift
            pred_aligned = batch_least_squares_alignment(pred_batch, gt_batch, mask_batch)
            
        pred_aligned.clamp_(min=MIN_EVAL_DEPTH, max=MAX_EVAL_DEPTH)

        # 4. 计算多区间误差 (Bin Evaluation)
        errors_batch = compute_errors_torch_bins(gt_batch, pred_aligned, mask_batch)
        all_errs_scene.append(errors_batch.cpu().numpy())
            
    if not all_errs_scene: return None
    return np.concatenate(all_errs_scene, axis=0)

def format_line(name, m, indent=0):
    sp = " " * indent
    m = np.nan_to_num(m)
    return "{:<50} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} |".format(
        sp + name[-50+indent:], m[0], m[1], m[2], m[3], m[4], m[5], m[6])

# =================================================================================
# 主循环
# =================================================================================

if __name__ == '__main__':
    if not torch.cuda.is_available(): exit("No CUDA device.")

    print(f"Starting Multi-Task Evaluation (Scale & Shift + Bins)")
    print(f"Total Tasks: {len(TASKS)}")

    for i, task in enumerate(TASKS):
        t_name = task['task_name']
        t_out = task['output_file']
        t_paths = task['paths']
        t_relative = task.get('use_relative_disparity_mode', True)

        print(f"\n[{i+1}/{len(TASKS)}] Processing Task: {t_name}")
        print(f"  Mode: {'Relative (1/x + Align)' if t_relative else 'Depth + Align'}")
        
        os.makedirs(os.path.dirname(t_out), exist_ok=True)
        
        results_map = {}
        all_errs_list = []
        
        for path_item in t_paths:
            s_name = path_item['name']
            p_dir = path_item['pred_dir']
            g_dir = path_item['gt_dir']
            
            if not os.path.isdir(p_dir):
                print(f"  [Error] Path missing: {p_dir}")
                continue

            # 计算该场景结果 [N, Bins, 8]
            scene_errs = evaluate_single_scene(p_dir, g_dir, s_name, is_relative_mode=t_relative)
            
            if scene_errs is not None:
                results_map[s_name] = scene_errs
                all_errs_list.append(scene_errs)

        # --- 生成报告 (模仿脚本1的结构) ---
        lines = []
        header = "{:<50} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |".format("Scene / Depth Bin", "AbsRel", "SqRel", "RMSE", "RMSElog", "a1", "a2", "a3")
        sep = "-" * len(header)
        
        lines += [
            f"Task: {t_name}", 
            f"Date: {datetime.now()}", 
            f"Align: Scale & Shift (Least Squares)",
            f"Input: {'Disparity' if t_relative else 'Depth'}",
            "="*100, 
            header, 
            sep
        ]
        
        # 1. 逐个场景输出
        for s_name in sorted(results_map.keys()):
            s_data = results_map[s_name] # [N, Bins, 8]
            s_sums = s_data.sum(axis=0)  # [Bins, 8]
            
            # Overall (Last index)
            overall_mean = compute_metrics_from_sums(s_sums[-1])
            lines.append(format_line(f"> {s_name} (All)", overall_mean))
            
            # Sub-bins
            for bin_idx, bin_label in enumerate(BIN_LABELS):
                bin_mean = compute_metrics_from_sums(s_sums[bin_idx])
                lines.append(format_line(f"   [{bin_label}]", bin_mean, indent=3))
            
            lines.append(sep)

        # 2. 总体输出
        if all_errs_list:
            total_data = np.concatenate(all_errs_list, axis=0) # [Total_N, Bins, 8]
            total_sums = total_data.sum(axis=0)
            
            lines.append("="*100)
            lines.append(">>> DATASET TOTAL AVERAGE <<<")
            lines.append(sep)
            
            # Total Overall
            total_mean = compute_metrics_from_sums(total_sums[-1])
            lines.append(format_line("TOTAL (All Depths)", total_mean))
            
            # Total Bins
            for bin_idx, bin_label in enumerate(BIN_LABELS):
                bin_mean = compute_metrics_from_sums(total_sums[bin_idx])
                lines.append(format_line(f"TOTAL [{bin_label}]", bin_mean))
            
            lines.append("="*100)

        with open(t_out, 'w') as f: f.write("\n".join(lines))
        print(f"  -> Report saved to {t_out}")
        
        del results_map, all_errs_list
        torch.cuda.empty_cache()

    print("\nAll tasks completed.")