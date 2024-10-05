import jieba
import jieba.analyse
import jieba.posseg as pseg
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.cluster import KMeans
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
def word_count(texts, freq_threshold=10):
    """
    统计词频，筛选高频词
    :param texts: 文本
    :param freq_threshold: 高频词阈值
    :return: 每一类词的高频词
    """
    # 分别存储动词、名词、形容词的词频
    verb_freq = {}
    noun_freq = {}
    adj_freq = {}

    for text in texts:
        words = pseg.cut(text)
        for word, flag in words:
            if flag.startswith('v'):  # 动词
                if word not in verb_freq:
                    verb_freq[word] = 0
                verb_freq[word] += 1
            elif flag.startswith('n'):  # 名词
                if word not in noun_freq:
                    noun_freq[word] = 0
                noun_freq[word] += 1
            elif flag.startswith('a'):  # 形容词
                if word not in adj_freq:
                    adj_freq[word] = 0
                adj_freq[word] += 1

    # 对每类词按照词频排序
    sorted_verbs = sorted(verb_freq.items(), key=lambda x: x[1], reverse=True)
    sorted_nouns = sorted(noun_freq.items(), key=lambda x: x[1], reverse=True)
    sorted_adjs = sorted(adj_freq.items(), key=lambda x: x[1], reverse=True)

    # 筛选出高频词
    high_freq_verbs = [word for word, count in sorted_verbs if count >= freq_threshold]
    high_freq_nouns = [word for word, count in sorted_nouns if count >= freq_threshold]
    high_freq_adjs = [word for word, count in sorted_adjs if count >= freq_threshold]

    return {
        'verbs': (sorted_verbs, high_freq_verbs),
        'nouns': (sorted_nouns, high_freq_nouns),
        'adjectives': (sorted_adjs, high_freq_adjs)
    }


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


