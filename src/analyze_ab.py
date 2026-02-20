# analyze_rfm.py
"""
RFM 用户分群分析脚本（最终优化版）
- 处理 Frequency 重复值导致的 qcut 错误（使用 pd.cut 兜底）
- 高价值用户退款/流失率对比 + 组统计表
- 生成 Frequency 分布图 + RFM 散点图
- 保存到 image 文件夹
- 兼容 VS Code 交互式运行（忽略 --f 参数）
"""

import sys
import argparse
import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ──── 配置日志 ────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 忽略无关警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──── 中文字体设置 ────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def get_base_dir():
    """兼容交互式运行和命令行"""
    default_dir = r"E:\项目集\电商"
    
    if 'ipykernel_launcher' in sys.argv[0] or any('kernel' in arg for arg in sys.argv):
        logger.info("检测到交互式环境，使用默认路径")
        return default_dir
    
    parser = argparse.ArgumentParser(description="RFM 用户分群分析")
    parser.add_argument('--base_dir', type=str, default=default_dir,
                        help="项目根目录")
    args, unknown = parser.parse_known_args()
    return args.base_dir

def load_data(base_dir: str) -> pd.DataFrame:
    data_dir = os.path.join(base_dir, "data", "clean")
    clean_path = os.path.join(data_dir, "clean_data.csv")
    
    if not os.path.exists(clean_path):
        logger.error(f"清洗数据文件不存在: {clean_path}")
        raise FileNotFoundError(clean_path)
    
    logger.info(f"读取清洗数据: {clean_path}")
    df = pd.read_csv(clean_path, encoding='utf-8-sig')
    
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    invalid = df['order_date'].isna().sum()
    if invalid > 0:
        logger.warning(f"有 {invalid} 条订单日期无效，已转为 NaT")
    
    df['net_sales'] = df['total_sales'] - df['refund_amount']
    
    # is_lost 兼容处理
    if 'order_status_sim' in df.columns:
        df['is_lost'] = ~df['order_status_sim'].isin(['Delivered', 'Returned'])
        logger.info("使用 order_status_sim 衍生 is_lost")
    else:
        logger.warning("未找到 order_status_sim，使用 is_lost = False")
        df['is_lost'] = False
    
    return df

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """计算 RFM 并打分"""
    current_date = df['order_date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('customer_id').agg({
        'order_date': lambda x: (current_date - x.max()).days,  # Recency
        'order_id': 'nunique',                                  # Frequency
        'net_sales': 'sum'                                      # Monetary
    }).rename(columns={'order_date': 'Recency', 'order_id': 'Frequency', 'net_sales': 'Monetary'})
    
    rfm = rfm[rfm['Monetary'] > 0].copy()
    
    logger.info("Frequency 分布（用于调试）：")
    print(rfm['Frequency'].value_counts().sort_index())
    
    # R 分数（越小越好，反序）
    rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
    
    # F 分数（处理重复值）
    try:
        rfm['F_score'] = pd.qcut(rfm['Frequency'], 4, labels=[1, 2, 3, 4], duplicates='drop')
    except ValueError as e:
        logger.warning(f"Frequency qcut 失败：{e}")
        logger.info("切换到 pd.cut 自定义区间")
        # 自定义区间：根据实际分布调整（你的数据 1-7）
        bins_f = [0, 1, 2, 4, rfm['Frequency'].max() + 1]  # 1次、2次、3-4次、5+次
        labels_f = [1, 2, 3, 4]
        rfm['F_score'] = pd.cut(rfm['Frequency'], bins=bins_f, labels=labels_f, include_lowest=True, right=False)
    
    # M 分数
    rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop')
    
    rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    
    return rfm

def analyze_rfm_insights(df: pd.DataFrame, rfm: pd.DataFrame, image_dir: str):
    """RFM 洞察 + 高价值用户分析 + Frequency 图"""
    # 高价值用户（R≥3, F≥3, M≥3）
    high_value = rfm[(rfm['R_score'] >= 3) & (rfm['F_score'] >= 3) & (rfm['M_score'] >= 3)]
    high_value_mask = df['customer_id'].isin(high_value.index)
    
    high_refund_rate = df[high_value_mask]['is_refunded'].mean() * 100 if high_value_mask.any() else 0
    avg_refund_rate = df['is_refunded'].mean() * 100
    
    high_lost_rate = df[high_value_mask]['is_lost'].mean() * 100 if high_value_mask.any() else 0
    avg_lost_rate = df['is_lost'].mean() * 100
    
    logger.info(f"高价值用户数量: {len(high_value)}")
    logger.info(f"高价值用户退款率: {high_refund_rate:.2f}% （整体平均 {avg_refund_rate:.2f}%）")
    logger.info(f"高价值用户履约前流失率: {high_lost_rate:.2f}% （整体平均 {avg_lost_rate:.2f}%）")
    
    # RFM 散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Frequency', y='Monetary', hue='R_score', size='Monetary',
        data=rfm, palette='viridis', sizes=(20, 200), alpha=0.7
    )
    plt.title('RFM 分群散点图（Monetary vs Frequency，按 R 分数着色）')
    plt.xlabel('购买频次 (Frequency)')
    plt.ylabel('净收入 (Monetary)')
    plt.legend(title='R 分数（越高越近期）')
    
    scatter_path = os.path.join(image_dir, 'rfm_scatter.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"RFM 散点图保存至: {scatter_path}")
    
    # Frequency 分布柱状图
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Frequency', data=rfm, palette='Blues_d',
                       order=range(1, int(rfm['Frequency'].max()) + 1))
    plt.title('用户购买频次分布', fontsize=14, fontweight='bold')
    plt.xlabel('购买次数 (Frequency)', fontsize=12)
    plt.ylabel('用户数量', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 柱子上方显示数值
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    freq_path = os.path.join(image_dir, 'frequency_distribution.png')
    plt.savefig(freq_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"购买频次分布图保存至: {freq_path}")

def main():
    base_dir = get_base_dir()
    image_dir = os.path.join(base_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    
    df = load_data(base_dir)
    rfm = compute_rfm(df)
    
    logger.info("\nRFM 基本统计：")
    print(rfm.describe())
    
    analyze_rfm_insights(df, rfm, image_dir)

if __name__ == "__main__":
    main()