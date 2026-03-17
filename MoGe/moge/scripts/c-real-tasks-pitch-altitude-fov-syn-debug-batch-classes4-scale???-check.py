import os
import logging
from tqdm import tqdm  # 如果没有安装，可以用 pip install tqdm，或者删掉相关代码

# ================= ⚙️ 配置区域 =================
BASE_DIR = "/data1/szq/TrainingData_Final_MoGe-All"  # 你的数据集根目录
INPUT_TXT = "train_index.txt"                         # 输入的索引文件
OUTPUT_TXT = "train_index_filtered.txt"               # 输出的过滤后文件
LOG_FILE = "filtering_process.log"                    # 日志文件路径
# =================================================

def setup_logger():
    """配置日志：同时输出到文件和终端"""
    logger = logging.getLogger("DatasetFilter")
    logger.setLevel(logging.INFO)
    
    # 清除旧的 handlers 防止重复打印
    if logger.handlers:
        logger.handlers = []

    # 1. 文件输出 (详细)
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. 终端输出 (简洁)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def main():
    logger = setup_logger()
    logger.info(f"🚀 开始任务")
    logger.info(f"📂 数据集根目录: {BASE_DIR}")
    logger.info(f"📄 输入列表文件: {INPUT_TXT}")
    
    # 1. 扫描硬盘上的真实文件夹
    if not os.path.exists(BASE_DIR):
        logger.error(f"❌ 错误: 找不到硬盘路径 {BASE_DIR}")
        return

    logger.info("正在扫描硬盘目录...")
    # 获取根目录下所有文件夹的名字 (比如 BC1, BC2, Artsci...)
    try:
        actual_folders = set(os.listdir(BASE_DIR))
    except Exception as e:
        logger.error(f"❌ 扫描目录失败: {e}")
        return
        
    logger.info(f"✅ 硬盘上共检测到 {len(actual_folders)} 个目录/文件")

    # 2. 读取 TXT
    try:
        with open(INPUT_TXT, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"❌ 错误: 找不到输入文件 {INPUT_TXT}")
        return

    kept_lines = []
    skipped_lines = []
    
    # 用于记录哪些场景缺失了 (去重统计)
    missing_scenes_stat = set()
    found_scenes_stat = set()

    logger.info(f"正在处理 {len(lines)} 行索引...")

    # 使用 tqdm 显示进度条 (如果是几百万行会很有用)
    for line in tqdm(lines, desc="Filtering", unit="lines"):
        content = line.strip()
        if not content:
            continue

        # 假设格式是 "BC1/000001" 或 "BC1/..."
        # 我们提取 "/" 前面的部分作为场景名
        parts = content.split('/')
        
        if len(parts) < 1:
            logger.warning(f"⚠️ 跳过格式错误的行: {content}")
            continue
            
        scene_name = parts[0] # 例如 "BC1"

        # 3. 核心判断
        if scene_name in actual_folders:
            kept_lines.append(line)
            found_scenes_stat.add(scene_name)
            # 如果你想 Log 每一行匹配成功 (会产生巨大日志，慎用):
            # logger.debug(f"匹配成功: {content}") 
        else:
            skipped_lines.append(content)
            missing_scenes_stat.add(scene_name)

    # 4. 写入结果
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.writelines(kept_lines)

    # ================= 📊 输出统计报告 =================
    logger.info("=" * 50)
    logger.info("📊 筛选结果统计报告")
    logger.info("=" * 50)
    logger.info(f"原始行数: {len(lines)}")
    logger.info(f"✅ 保留行数: {len(kept_lines)} (已写入 {OUTPUT_TXT})")
    logger.info(f"❌ 删除行数: {len(skipped_lines)}")
    logger.info("-" * 30)
    
    # 报告具体的场景匹配情况
    logger.info(f"硬盘存在的场景数: {len(found_scenes_stat)}")
    logger.info(f"索引中缺失的场景数: {len(missing_scenes_stat)}")

    if missing_scenes_stat:
        logger.info("\n⚠️ 以下场景在 txt 中存在，但在硬盘上【找不到】，相关行已被删除：")
        logger.info("-" * 30)
        for missing_scene in sorted(list(missing_scenes_stat)):
            # 统计这个场景删了多少行
            count = sum(1 for line in skipped_lines if line.startswith(missing_scene + "/"))
            logger.info(f"   [MISSING] {missing_scene:<20} (删除了 {count} 行)")
            
        logger.info("-" * 30)
        logger.info(f"详细删除明细请查看日志文件: {LOG_FILE}")
        
        # 在 Log 文件里记录每一行被删的内容（不在终端刷屏，但在文件里记下来）
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        logger.info("\n=== 删除内容详情 (Detailed Removed Lines) ===")
        for item in skipped_lines:
            logger.info(f"Deleted: {item} (Reason: Folder '{item.split('/')[0]}' not found)")
    else:
        logger.info("🎉 完美！TXT 中的所有场景都在硬盘上找到了。")

if __name__ == "__main__":
    main()