import pandas as pd
import io

# 1. 模拟你的数据 (实际使用时请改用 pd.read_csv('你的文件名.csv'))


# 读取数据
df = pd.read_csv('/home/szq/moge2/final_merged.csv')
# df = pd.read_csv(io.StringIO(data))

# 2. 执行统计
# value_counts() 会自动统计 Split 列中每个唯一值出现的次数
stats = df['Split'].value_counts(dropna=False)

# 3. 打印结果
print("--- 统计结果 ---")
print(stats)

# 4. 如果你想针对性地获取某个值（防止某些值为 0）
train_count = (df['Split'] == 'train').sum()
nan_count = (df['Split'] == 'NAN').sum() # 这里假设 NAN 是字符串
val_count = (df['Split'] == 'val').sum()

print(f"\n具体数值明细:")
print(f"Train 条目: {train_count}")
print(f"NAN 条目: {nan_count}")
print(f"Val 条目: {val_count}")