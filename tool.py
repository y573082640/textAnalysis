import pandas as pd
import re
import jieba

# 预定义jieba相关内容
# 加载自定义词典
zmjh_path="resources/zmjh.txt"
with open(zmjh_path, 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()
    for w in stopwords:
        jieba.add_word(w)  
stopword_path="resources/stopwords.txt"
tokenizer = jieba.Tokenizer()

def remove_stopwords(text):
    """
    移除文本中的停用词

    参数:
        texts (list of str): 输入的文本列表
        stopword_path: 停用词文件路径

    返回:
        list of str: 移除停用词后的文本列表
    """
    # 读入stopword_path文件
    with open(stopword_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()

    processed_text = []
    
    # 使用jieba分词
    words = tokenizer.lcut(text)
    # 过滤停用词
    filtered_words = [word for word in words if word not in stopwords]
    # 将过滤后的单词组合成句子
    processed_text = ''.join(filtered_words)

    return processed_text

def preprocess_with_glm(texts,prompt):
    prompt = """
        你是一个文本预处理专家，任务是对输入文本进行以下处理：
        1. 将文本中的所有链接替换为 "[link]"，所有图片替换为 "[image]"。
        2. 如果文本涉及多篇日记/多个日期，用“>>>>>>>>>>”进行日记分割。
        3. 移除不必要的段落标记，并根据段落标记将段落拆分。
        4. 最终输出格式为 JSON，格式如下：
        [
            {date:"", content:["para1", "para2", ...]},
            {date:"", content:["para1", "para2", ...]},
            ...
        ]
        注意：
        - 每篇日记内容必须保存在 content 字段中，以列表形式列出每个段落。
        - 处理后的每篇日记内容必须保持原文的格式，不要进行任何修改。
    """


def export_topic_keywords(topic_model, file_name="topic_keywords.csv"):
    """
    导出每个主题的关键词到 CSV 文件。

    参数：
    topic_model (BERTopic): 训练好的 BERTopic 模型。
    file_name (str): 导出的 CSV 文件名。
    """
    # 获取所有主题的信息
    topic_info = topic_model.get_topic_info()

    # 保存为 CSV 文件
    topic_info.to_csv(file_name, index=False)
    print(f"主题关键词已导出到 {file_name}")

def save_visualizations(topic_model, topics_file="topics.html", barchart_file="barchart.html"):
    """
    保存 BERTopic 模型的可视化结果。

    参数：
    topic_model (BERTopic): 训练好的 BERTopic 模型。
    topics_file (str): 保存主题可视化结果的文件名（HTML格式）。
    barchart_file (str): 保存词频柱状图的文件名（HTML格式）。
    """
    # 保存主题的可视化
    topics_visualization = topic_model.visualize_topics()
    topics_visualization.write_html(topics_file)
    print(f"主题可视化已保存到 {topics_file}")

    # 保存词频柱状图
    barchart_visualization = topic_model.visualize_barchart()
    barchart_visualization.write_html(barchart_file)
    print(f"词频柱状图已保存到 {barchart_file}")


def save_topic_model(topic_model, file_name="bertopic_model"):
    """
    保存 BERTopic 模型到指定文件。

    参数：
    topic_model (BERTopic): 训练好的 BERTopic 模型。
    file_name (str): 模型保存的文件名，不需要扩展名（会保存为 `.pkl` 文件）。
    """
    topic_model.save(file_name)
    print(f"模型已保存到 {file_name}.pkl")


def export_topic_results(texts, topics, probabilities, file_name="topic_results.csv"):
    """
    将文本的主题分配结果导出为 CSV 文件。

    参数：
    texts (list of str): 文本数据。
    topics (list of int): 每个文档的主题编号。
    probabilities (list of list): 每个文档的主题概率分布。
    file_name (str): 导出的 CSV 文件名。
    """
    # 创建 DataFrame 存储结果
    df = pd.DataFrame({
        'Text': texts,
        'Topic': topics,
        'Probabilities': probabilities
    })

    # 导出为 CSV 文件
    df.to_csv(file_name, index=False)
    print(f"主题结果已导出到 {file_name}")


def preprocess(text):
    """
    移除文本中的HTML标签

    参数:
        text (str): 输入的文本

    返回:
        str: 移除HTML标签后的文本
    """
    # 使用正则表达式匹配并替换所有 <!-- wp:image --> 块
    cleaned_text = re.sub(r'<!-- wp:image.*?<!-- /wp:image -->', '[image]', text, flags=re.DOTALL)

    # 继续移除其他 HTML 标签
    cleaned_text = re.sub(r'<[^>]*>', '\n', cleaned_text)

    # 替换特殊字符
    cleaned_text = cleaned_text.replace("&nbsp;", " ")

    # 加在一起
    cleaned_text = "\n".join([line for line in cleaned_text.splitlines() if line.strip()])

    # 移除停用词
    cleaned_text = remove_stopwords(cleaned_text)

    return cleaned_text


if __name__ == '__main__':
    preprocess("""
    <!-- wp:paragraph -->
<p>疲倦的同时看着孩子们纯真的笑容，这次支教的意义越来越清晰，给他们带来哥哥姐姐们的温暖，给他们带来不一样的世界，能给在大山里的他们一次看见世界的机会。</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>邹稚媛的田野日记</p>
<!-- /wp:paragraph -->
""")