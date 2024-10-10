import jieba
import jieba.analyse
import jieba.posseg as pseg
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import Counter
from itertools import combinations

def co_occurrence_analysis(texts, window_size=12, freq_threshold=2, output_file="output/co_occurrence_patterns.xlsx"):
    """
    共现分析，找到频繁共现的词语对，并将结果保存到 Excel 文件中
    :param texts: 文本列表
    :param window_size: 窗口大小，表示在同一窗口内出现的词语对会被认为共现
    :param freq_threshold: 共现频率阈值，筛选频繁共现词对
    :param output_file: 输出的 Excel 文件名
    """
    # 存储所有共现词对的频次
    co_occurrence_freq = Counter()

    for text in texts:
        words = [word for word, flag in pseg.cut(text)]  # 对文本进行分词
        # 滑动窗口获取共现词对
        for i in range(len(words) - window_size + 1):
            window_words = words[i:i + window_size]
            word_pairs = combinations(window_words, 2)
            co_occurrence_freq.update(word_pairs)

    # 筛选出频繁共现的词对
    frequent_pairs = {pair: count for pair, count in co_occurrence_freq.items() if count >= freq_threshold}

    # 将结果转为 DataFrame
    word1_list = []
    word2_list = []
    counts = []
    for (word1, word2), count in frequent_pairs.items():
        word1_list.append(word1)
        word2_list.append(word2)
        counts.append(count)

    df = pd.DataFrame({
        'Word 1': word1_list,
        'Word 2': word2_list,
        'Co-occurrence Frequency': counts
    })

    # 保存到 Excel 文件
    df.to_excel(output_file, index=False)
    print(f"共现分析结果已保存到 {output_file}")

    return df

def word_count(texts, freq_threshold=5, output_file="output/word_frequencies.xlsx"):
    """
    统计所有词的词频，并将结果保存到 Excel 文件中
    :param texts: 文本列表
    :param freq_threshold: 高频词阈值，默认为1，统计所有词
    :param output_file: 输出的 Excel 文件名
    """
    pos_mapping = {
        'a': '形容词',
        'ad': '副形词',
        'an': '名形词',
        'b': '区别词',
        'c': '连词',
        'd': '副词',
        'e': '叹词',
        'f': '方位词',
        'g': '语素',
        'h': '前接成分',
        'i': '成语',
        'j': '简称略语',
        'k': '后接成分',
        'l': '习用语',
        'm': '数词',
        'mq': '数量词',
        'n': '名词',
        'nr': '人名',
        'nrfg': '古代人名',
        'nrt': '音译人名',
        'ns': '地名',
        'nt': '机构团体',
        'nz': '其他专名',
        'o': '拟声词',
        'p': '介词',
        'q': '量词',
        'r': '代词',
        'rz': '指示代词',
        's': '处所词',
        't': '时间词',
        'tg': '时间语素',
        'u': '助词',
        'ud': '结构助词',
        'ug': '时态助词',
        'uz': '着词助词',
        'uv': '连词助词',
        'v': '动词',
        'vd': '副动词',
        'vn': '名动词',
        'vg': '动词语素',
        'w': '标点符号',
        'x': '非语素字',
        'y': '语气词',
        'z': '状态词',
        'zg': '状态语素',
        'eng': '外来语'
    }
    # 存储所有词的词频
    word_freq = {}

    for text in texts:
        words = pseg.cut(text)
        for word, flag in words:
            if (word, flag) not in word_freq:
                word_freq[(word, flag)] = 0
            word_freq[(word, flag)] += 1

    # 词频结果转为 DataFrame
    all_words = []
    all_flags = []
    all_counts = []

    for (word, flag), count in word_freq.items():
        if count >= freq_threshold:
            all_words.append(word)
            all_flags.append(pos_mapping.get(flag,flag))
            all_counts.append(count)

    # 创建 DataFrame
    df = pd.DataFrame({
        '词语': all_words,
        '词性': all_flags,
        '词频': all_counts
    })

    # 将结果保存为 Excel 文件
    df.to_excel(output_file, index=False)
    print(f"词频结果已保存到 {output_file}")

    return df

def keyword_extraction(texts, top_k=5):
    """
    提取每段文本中的前 top_k 个关键词。

    参数：
    texts (list of str): 输入的文本列表。
    top_k (int): 每个文本中要提取的关键词数量。

    返回：
    keywords_list (list of list): 每个文本对应的关键词列表。
    """
    keywords_list = []

    for text in texts:
        # 使用TF-IDF算法提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=top_k)
        keywords_list.append(keywords)

    return keywords_list

def topic_modeling(texts, num_components=10, n_neighbors=15, min_dist=0.1):
    """
    使用 BERTopic 对文本集合进行主题建模，包含降维优化。

    参数：
    texts (list of str): 输入的文本列表。
    num_components (int): UMAP 降维的目标维度数，默认为 10。
    n_neighbors (int): UMAP 的邻居参数，默认为 15。
    min_dist (float): UMAP 中最小距离参数，控制降维的紧密性，默认为 0.1。

    返回：
    topics (list of int): 每个文档的主题编号。
    probabilities (list of list): 每个文档的主题概率分布。
    topic_model (BERTopic): 训练好的 BERTopic 模型，用于后续分析。
    """
    # 使用 TF-IDF 对文本进行向量化表示
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(texts)

    # 使用 UMAP 对高维数据进行降维处理
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=num_components, min_dist=min_dist, random_state=42)

    # 初始化 BERTopic 模型，传入自定义的 UMAP 模型
    topic_model = BERTopic(umap_model=umap_model)

    # 使用 BERTopic 进行主题建模
    topics, probabilities = topic_model.fit_transform(X)

    return topics, probabilities, topic_model

def entiment_analysis(texts):
    """
    使用基于 GoEmotions 的多类别情感分析模型对文本列表进行情感分析。

    参数：
    texts (list of str): 输入的文本列表。

    返回：
    results (list of dict): 每个文本对应的情感分析结果，包括情感标签和分数。
    """
    # 加载 GoEmotions 模型和分词器
    model_name = "uer/roberta-base-finetuned-chinese-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # 创建情感分析管道
    sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    # 对每个文本进行多类别情感分析
    results = sentiment_pipeline(texts)

    return results

def text_clustering(texts, num_clusters=5):
    """
    使用 TF-IDF 和 KMeans 对文本列表进行聚类。

    参数：
    texts (list of str): 输入的文本列表。
    num_clusters (int): 聚类数量，默认为5。

    返回：
    clusters (list of int): 每个文本的聚类标签。
    kmeans (KMeans): 训练好的 KMeans 模型，用于进一步分析。
    """
    # 将文本转换为 TF-IDF 特征向量
    vectorizer = TfidfVectorizer(stop_words='chinese')
    X = vectorizer.fit_transform(texts)

    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # 获取每个文本的聚类标签
    clusters = kmeans.labels_

    return clusters, kmeans

def text_classification(texts):
    ...

def named_entity_recognition(texts):
    ...

def knowledgemap_generation(texts):
    ...


