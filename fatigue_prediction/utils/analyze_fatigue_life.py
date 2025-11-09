import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('/home/zty/code/mental_fatigue/dataset/data_all_strain-controlled.csv')

# 从文件名中提取材料名
def extract_material(filename):
    # 常见材料名列表
    known_materials = [
        "1Cr18Ni9T", "16MnR", "45mild", "304SS", "410", "1045HR", 
        "2024-T3", "6061-T6", "7075-T651", "AISI 316L", "Al5083", 
        "AZ31B", "AZ61A", "BT9", "CA", "CP-Ti", "CS", "CuZn37", 
        "E235", "E355", "GH4169", "Haynes188", "HRB335", "inconel718", 
        "mild", "PA38", "pureTi", "q235b", "S45C", "S347", "S460N", 
        "SNCM630", "TC4", "X5CrNi", "ZK60", "5% chrome work roll steel",
        "30CrMnSiA", "2024-STSA", "2024-T3", "2198-T8", "6082-T6",
        "7075-T651", "LY12CZ", "SM45C"
    ]
    
    # 检查文件名是否以任何已知材料名称开头
    for material in known_materials:
        if filename.startswith(material):
            return material
    
    # 如果没有匹配到预定义材料，提取文件名中第一个连字符前的部分
    match = re.match(r'^([^-]+)', filename)
    if match:
        return match.group(1)
    
    return "Unknown"

# 应用提取函数并创建新列
df['Material'] = df['load'].apply(extract_material)

# 计算每种材料的疲劳寿命统计
material_stats = df.groupby('Material')['Nf(label)'].agg(['min', 'max', 'mean', 'std', 'count']).reset_index()
material_stats = material_stats.sort_values('mean', ascending=False)

# 打印统计结果
print("\n材料疲劳寿命统计分析")
print("=" * 80)
print(f"{'材料':15} {'最小值':10} {'最大值':10} {'均值':10} {'标准差':10} {'样本数':8}")
print("-" * 80)

for _, row in material_stats.iterrows():
    print(f"{row['Material']:15} {row['min']:10.4f} {row['max']:10.4f} {row['mean']:10.4f} {row['std']:10.4f} {row['count']:8}")

# 计算总体统计
total_min = df['Nf(label)'].min()
total_max = df['Nf(label)'].max()
total_mean = df['Nf(label)'].mean()
total_std = df['Nf(label)'].std()

print("\n总体疲劳寿命统计")
print("=" * 80)
print(f"最小值: {total_min:.4f}")
print(f"最大值: {total_max:.4f}")
print(f"均值: {total_mean:.4f}")
print(f"标准差: {total_std:.4f}")
print(f"总样本数: {len(df)}")

# 可视化 - 各材料的平均疲劳寿命柱状图（仅显示样本数大于5的材料）
plt.figure(figsize=(14, 8))
filtered_stats = material_stats[material_stats['count'] > 5].sort_values('mean')
plt.bar(filtered_stats['Material'], filtered_stats['mean'], yerr=filtered_stats['std'])
plt.xlabel('Material')
plt.ylabel('Mean Fatigue Life (log)')
plt.title('Mean Fatigue Life of Materials')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('./utils/material_fatigue_life.png')

# 可视化 - 疲劳寿命分布直方图
plt.figure(figsize=(10, 6))
plt.hist(df['Nf(label)'], bins=20, alpha=0.7, color='blue')
plt.xlabel('Fatigue Life (log)')
plt.ylabel('Frequency')
plt.title('Fatigue Life Distribution')
plt.savefig('./utils/fatigue_life_distribution.png')

print("\n已生成图表：")
print("1. material_fatigue_life.png - 各材料的平均疲劳寿命柱状图")
print("2. fatigue_life_distribution.png - 疲劳寿命分布直方图") 