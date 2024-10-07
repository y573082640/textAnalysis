from tool import *
# from algorithm import *
import pandas as pd

def run_pipeline(file_path):
    # 读取csv文件
    df = pd.read_excel(file_path)
    # 调用preprocess函数处理content列
    df['content_preprocessed'] = df['Content'].apply(preprocess)
    # 返回处理后的数据框
    print(df.head())
    return df

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    file_path = "resources/壮苗小站文章-Export-2024-October-05-0752.xlsx"
    run_pipeline(file_path)
