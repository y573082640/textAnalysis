from tool import *
from algorithm import *
import pandas as pd
import json
def run_pipeline(file_path):
    # 读取csv文件
    df = pd.read_excel(file_path)

    # 调用preprocess函数处理content列
    df['content_preprocessed'] = df['Content'].apply(preprocess)

    # 词频统计
    word_count(df['content_preprocessed'].tolist())
    co_occurrence_analysis(df['content_preprocessed'].tolist())
    # 保存处理后的数据框到新的csv文件
    df.to_excel('output/content_preprocessed.xlsx', index=False)

    return df

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    file_path = "resources/壮苗小站文章-Export-2024-October-05-0752.xlsx"
    run_pipeline(file_path)
