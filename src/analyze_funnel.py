# analyze_funnel.py
"""
订单漏斗分析脚本（最终优化版）
- 兼容 Jupyter/VS Code 交互式运行（忽略 --f 参数）
- 支持命令行参数 --base_dir
- 衍生 order_status_sim
- 双漏斗（原始 & 模拟） + 品类洞察
- 所有图表保存到 image 文件夹
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
    """兼容交互式运行（VS Code / Jupyter）和命令行"""
    default_dir = r"E:\项目集\电商"
    
    # 如果是从 VS Code 交互式或 Jupyter 运行，忽略额外参数
    if 'ipykernel_launcher' in sys.argv[0] or any('kernel' in arg for arg in sys.argv):
        logger.info("检测到交互式环境，使用默认路径")
        return default_dir
    
    # 正常命令行运行，解析参数
    parser = argparse.ArgumentParser(description="订单漏斗分析脚本")
    parser.add_argument('--base_dir', type=str, default=default_dir,
                        help="项目根目录")
    args, unknown = parser.parse_known_args()  # 忽略未知参数（如 --f=xxx.json）
    return args.base_dir

def load_data(base_dir: str) -> pd.DataFrame:
    data_dir = os.path.join(base_dir, "data", "clean")
    clean_path = os.path.join(data_dir, "clean_data.csv")
    
    if not os.path.exists(clean_path):
        logger.error(f"清洗数据文件不存在: {clean_path}")
        raise FileNotFoundError(clean_path)
    
    logger.info(f"读取清洗数据: {clean_path}")
    df = pd.read_csv(clean_path, encoding='utf-8-sig')
    
    # 日期转换
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    invalid = df['order_date'].isna().sum()
    if invalid > 0:
        logger.warning(f"有 {invalid} 条订单日期无效，已转为 NaT")
    
    # 计算净销售额
    df['net_sales'] = df['total_sales'] - df['refund_amount']
    
    return df

def derive_order_status_sim(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("开始衍生 order_status_sim...")
    
    logger.info("原始 order_status 分布：")
    print(df['order_status'].value_counts(dropna=False))
    
    df['order_status_sim'] = 'Delivered'
    df.loc[df['is_refunded'] == 1, 'order_status_sim'] = 'Returned'
    
    np.random.seed(42)
    non_delivered_ratio = 0.08
    non_delivered_idx = df[df['order_status_sim'] == 'Delivered'].sample(frac=non_delivered_ratio).index
    
    statuses = np.random.choice(['Pending', 'Shipped', 'Cancelled'], 
                                size=len(non_delivered_idx), 
                                p=[0.4, 0.4, 0.2])
    df.loc[non_delivered_idx, 'order_status_sim'] = statuses
    
    logger.info("模拟后 order_status_sim 分布：")
    print(df['order_status_sim'].value_counts(normalize=True) * 100)
    
    return df

def compute_funnel(df: pd.DataFrame, status_col: str = 'order_status_sim') -> pd.DataFrame:
    df['month'] = df['order_date'].dt.to_period('M')
    
    funnel = df.groupby('month').agg(
        Placed=('order_id', 'nunique'),
        Shipped=(status_col, lambda x: x.isin(['Shipped', 'Delivered', 'Returned']).sum()),
        Delivered=(status_col, lambda x: x.isin(['Delivered', 'Returned']).sum()),
        Refunded=(status_col, lambda x: (x == 'Returned').sum()),
        Net_Revenue=('net_sales', 'sum')
    )
    
    funnel['Ship Rate (%)'] = (funnel['Shipped'] / funnel['Placed'].replace(0, 1) * 100).round(2)
    funnel['Deliver Rate (%)'] = (funnel['Delivered'] / funnel['Shipped'].replace(0, 1) * 100).round(2)
    funnel['Refund Rate (%)'] = (funnel['Refunded'] / funnel['Delivered'].replace(0, 1) * 100).round(2)
    
    return funnel

def plot_funnel(overall: pd.Series, stages: list, title: str, save_name: str, image_dir: str):
    values = overall[stages]
    total = values[0]
    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=stages, y=values, hue=stages, palette='viridis', legend=False)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('漏斗阶段', fontsize=12)
    plt.ylabel('订单数量', fontsize=12)
    
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + total*0.01,
                 f"{int(height)}\n({height/total*100:.1f}%)",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(image_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"{title} 已保存至: {save_path}")

def analyze_category_insights(df: pd.DataFrame, image_dir: str):
    df['is_lost'] = ~df['order_status_sim'].isin(['Delivered', 'Returned'])
    
    insights = df.groupby('category').agg({
        'order_id': 'nunique',
        'is_refunded': 'sum',
        'is_lost': 'sum',
        'net_sales': 'sum'
    }).rename(columns={'order_id': 'Total', 'is_refunded': 'Refunded', 
                       'is_lost': 'Lost Before Delivery', 'net_sales': 'Net Revenue'})
    
    insights['Refund Rate (%)'] = (insights['Refunded'] / insights['Total'] * 100).round(2)
    insights['Lost Rate (%)'] = (insights['Lost Before Delivery'] / insights['Total'] * 100).round(2)
    
    logger.info("\n各品类退款 & 履约流失率：")
    print(insights.sort_values('Refund Rate (%)', ascending=False))
    
    # 退款率图
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Refund Rate (%)', y=insights.index, data=insights.reset_index(),
                hue=insights.index, palette='Blues_d', legend=False)
    plt.title('各品类退款率对比')
    plt.xlabel('退款率 (%)')
    plt.ylabel('品类')
    plt.savefig(os.path.join(image_dir, '品类退款率.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 履约前流失率图
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Lost Rate (%)', y=insights.index, data=insights.reset_index(),
                hue=insights.index, palette='Reds_d', legend=False)
    plt.title('各品类履约前流失率对比')
    plt.xlabel('履约前流失率 (%)')
    plt.ylabel('品类')
    plt.savefig(os.path.join(image_dir, '品类履约前流失率.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_dir = get_base_dir()
    image_dir = os.path.join(base_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    
    df = load_data(base_dir)
    df = derive_order_status_sim(df)
    
    # 原始漏斗（对比用）
    funnel_orig = compute_funnel(df, status_col='order_status')
    logger.info("\n原始月度漏斗转化率：")
    print(funnel_orig)
    
    overall_orig = funnel_orig.mean(numeric_only=True).round(2)
    plot_funnel(overall_orig, ['Placed', 'Shipped', 'Delivered', 'Refunded'],
                '整体订单漏斗（原始状态）', '整体订单漏斗.png', image_dir)
    
    # 模拟漏斗
    funnel_sim = compute_funnel(df, status_col='order_status_sim')
    logger.info("\n模拟后月度漏斗转化率：")
    print(funnel_sim)
    
    overall_sim = funnel_sim.mean(numeric_only=True).round(2)
    plot_funnel(overall_sim, ['Placed', 'Shipped', 'Delivered', 'Refunded'],
                '模拟后整体订单漏斗（含衍生状态）', '模拟订单漏斗.png', image_dir)
    
    # 品类洞察
    analyze_category_insights(df, image_dir)

if __name__ == "__main__":
    main()