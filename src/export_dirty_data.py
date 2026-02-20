# export_dirty_data.py
"""
功能：从指定的原始干净数据文件开始，注入模拟脏数据（刷单、退款、负运费等），
然后导出为 dirty_data.csv，供后续清洗模块使用。
"""

import pandas as pd
import numpy as np
import os
import logging

# 配置日志，方便追踪执行过程
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():

    input_path = r"E:\项目集\电商\data\raw\amazon_sales_dataset - 副本.csv"
    
    output_path = r"E:\项目集\电商\data\dirty\dirty_data.csv"

    # 检查文件是否存在
    if not os.path.exists(input_path):
        logger.error(f"原始文件不存在: {input_path}")
        logger.info("当前工作目录下的文件列表：")
        try:
            logger.info("\n".join(os.listdir(os.path.dirname(input_path))))
        except:
            logger.info("无法列出目录内容")
        return

    logger.info(f"正在读取原始数据: {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning("UTF-8 编码失败，尝试 GBK 编码...")
        df = pd.read_csv(input_path, encoding='gbk')
    except Exception as e:
        logger.error(f"读取文件失败: {str(e)}")
        return

    logger.info(f"原始数据形状: {df.shape}")
    if 'quantity' not in df.columns or 'total_sales' not in df.columns:
        logger.error("数据缺少关键字段 'quantity' 或 'total_sales'，无法继续")
        return

    # 注入脏数据

    np.random.seed(42)  # 固定种子，保证可复现
    df_dirty = df.copy()

    logger.info("开始注入模拟脏数据...")

    # 1. 刷单异常（约3%订单）
    fraud_ratio = 0.03
    fraud_count = int(len(df_dirty) * fraud_ratio)
    fraud_idx = np.random.choice(df_dirty.index, size=fraud_count, replace=False)
    multiplier = np.random.uniform(20, 100, len(fraud_idx))
    df_dirty.loc[fraud_idx, 'quantity'] *= multiplier

    # 同步更新 total_sales
    if 'unit_price' in df_dirty.columns and 'discount' in df_dirty.columns:
        df_dirty.loc[fraud_idx, 'total_sales'] = (
            df_dirty.loc[fraud_idx, 'quantity'] *
            df_dirty.loc[fraud_idx, 'unit_price'] *
            (1 - df_dirty.loc[fraud_idx, 'discount'])
        )
    logger.info(f"注入刷单异常订单: {len(fraud_idx)} 条")

    # 2. 退款模拟（约8%订单）
    refund_ratio = 0.08
    refund_count = int(len(df_dirty) * refund_ratio)
    refund_idx = np.random.choice(df_dirty.index, size=refund_count, replace=False)

    df_dirty['is_refunded'] = 0
    df_dirty.loc[refund_idx, 'is_refunded'] = 1

    df_dirty['refund_amount'] = 0.0
    df_dirty.loc[refund_idx, 'refund_amount'] = (
        df_dirty.loc[refund_idx, 'total_sales'] *
        np.random.uniform(0.8, 1.0, len(refund_idx))
    )
    logger.info(f"注入退款订单: {len(refund_idx)} 条")

    # 3. 负运费（约5%订单）
    neg_ratio = 0.05
    neg_count = int(len(df_dirty) * neg_ratio)
    neg_idx = np.random.choice(df_dirty.index, size=neg_count, replace=False)
    df_dirty.loc[neg_idx, 'shipping_cost'] *= -1
    logger.info(f"注入负运费订单: {len(neg_idx)} 条")

    
    # 导出
    try:
        df_dirty.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"脏数据导出成功 → {os.path.abspath(output_path)}")
        logger.info(f"导出数据行数: {len(df_dirty):,}")
        logger.info(f"文件大小约: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        logger.error(f"导出失败: {str(e)}")


if __name__ == "__main__":
    main()