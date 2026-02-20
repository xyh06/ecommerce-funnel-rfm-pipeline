# clean_data.py
"""
功能：读取上游脏数据 dirty_data.csv，进行工程化清洗，
输出清洗后的 clean_data.csv
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dirty_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"脏数据文件不存在: {input_path}")
    
    logger.info(f"读取脏数据: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    logger.info(f"脏数据形状: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    logger.info("开始清洗...")

    # 1. 处理负值（shipping_cost < 0 → 设为 0 或中位数）
    if 'shipping_cost' in df_clean.columns:
        neg_count = (df_clean['shipping_cost'] < 0).sum()
        if neg_count > 0:
            median_shipping = df_clean['shipping_cost'][df_clean['shipping_cost'] > 0].median()
            df_clean.loc[df_clean['shipping_cost'] < 0, 'shipping_cost'] = median_shipping
            logger.info(f"修正负运费记录: {neg_count} 条 → 用中位数 {median_shipping:.2f} 填充")

    # 2. 异常值处理（quantity & total_sales） - IQR法 + 业务阈值
    for col in ['quantity', 'total_sales']:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            upper = Q3 + 1.5 * IQR
            outlier_count = (df_clean[col] > upper).sum()
            
            # 额外业务规则：单笔 quantity > 50 或 total_sales > 1e6 视为异常
            business_outlier = (df_clean[col] > 50) if col == 'quantity' else (df_clean[col] > 1000000)
            total_outlier = ((df_clean[col] > upper) | business_outlier).sum()
            
            df_clean.loc[df_clean[col] > upper, col] = upper  # cap 而非 drop，保留更多信息
            logger.info(f"{col} 异常值（IQR + 业务规则）: {total_outlier} 条，已 cap 到 {upper:.2f}")

    # 3. 修复 total_sales 计算不一致（如果有 unit_price 和 discount）
    if all(col in df_clean.columns for col in ['quantity', 'unit_price', 'discount', 'total_sales']):
        computed = df_clean['quantity'] * df_clean['unit_price'] * (1 - df_clean['discount'])
        diff_mask = abs(df_clean['total_sales'] - computed) > 1  # 容忍 1 元误差
        inconsistent_count = diff_mask.sum()
        if inconsistent_count > 0:
            df_clean.loc[diff_mask, 'total_sales'] = computed[diff_mask]
            logger.info(f"修复 total_sales 计算不一致: {inconsistent_count} 条")

    # 4. 退款金额校验（is_refunded=1 但 refund_amount=0 的补齐）
    if all(col in df_clean.columns for col in ['is_refunded', 'refund_amount', 'total_sales']):
        invalid_refund = (df_clean['is_refunded'] == 1) & (df_clean['refund_amount'] == 0)
        invalid_count = invalid_refund.sum()
        if invalid_count > 0:
            df_clean.loc[invalid_refund, 'refund_amount'] = df_clean.loc[invalid_refund, 'total_sales'] * 0.9
            logger.info(f"补齐无效退款金额: {invalid_count} 条（设为原价 90%）")
    
    # 5. 最终检查
    logger.info("清洗完成")
    logger.info(f"清洗后形状: {df_clean.shape}")
    return df_clean


def export_clean_data(df_clean: pd.DataFrame, output_path: str):
    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"清洗数据导出成功: {os.path.abspath(output_path)}")
    logger.info(f"文件大小约: {os.path.getsize(output_path)/1024/1024:.2f} MB")


def main():
    dirty_path = r"E:\项目集\电商\data\dirty\dirty_data.csv"
    clean_path = r"E:\项目集\电商\data\clean\clean_data.csv"

    try:
        df_dirty = load_dirty_data(dirty_path)
        df_clean = clean_data(df_dirty)
        export_clean_data(df_clean, clean_path)
    except Exception as e:
        logger.error(f"清洗流程失败: {str(e)}")


if __name__ == "__main__":
    main()