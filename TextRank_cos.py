# Алгоритм TextRank с косинусной мерой схожести
from itertools import combinations
from nltk.stem.snowball import RussianStemmer
from Preprocessing import sent_tokenizer, word_tokenizer, stop_words
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm


class StemTokenizer(object):
    def __init__(self):
         self.st = RussianStemmer()

    def __call__(self, doc):
        return [self.st.stem(t) for t in word_tokenizer.tokenize(doc.lower())]


def tfidf(text):  # Возвращает матрицу tfidf по тексту с учетом стемминга и стоп-слов

    sentences = sent_tokenizer(text)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words, tokenizer=StemTokenizer()).fit_transform(sentences)
    a = tfidf.toarray()
    return a


def similarity(s1, s2):
    """
    :param s1: Вектор tfidf предложения s1
    :param s2: Вектор tfidf предложения s2
    :return float: Степень схожести предложений
    """
    n1 = norm(s1)
    n2 = norm(s2)
    if not n1 or not n2:
        return 0.0
    return dot(s1, s2)/(n1*n2)


def text_rank(text):
    """
    :param text: Исходный текст
    :return list: Список троек (номер предложения i, ранг этого предложения pr[i], предложение s),
             отсортированный по убыванию ранга
    """
    sentences = sent_tokenizer(text)  # Список предложений текста
    if len(sentences) < 2:
        s = sentences[0]
        return [(1, 0, s)]

    # Нумерация предложений с 0 до len(sent)-1 и генерация всех возможных комбинаций по два:
    a = tfidf(text)
    pairs = combinations(range(len(sentences)), 2)
    scores = [(i, j, similarity(a[i, :], a[j, :])) for i, j in pairs]
    scores = filter(lambda x: x[2], scores)  # Фильтр совсем непохожих предложений

    g = nx.Graph()
    g.add_weighted_edges_from(scores)  # Создание графа с весами ребер
    pr = nx.pagerank(g)  # Словарь: ключ - номер вершины, значение - её ранг

    return sorted(((i, pr[i], s) for i, s in enumerate(sentences) if i in pr),
                  key=lambda x: pr[x[0]], reverse=True)  # Сортировка по убыванию ранга тройки


def text_rank_extract(text, n=5):
    tr = text_rank(text)
    top_n = sorted(tr[:n])  # Сортировка первых n предложений по их порядку в тексте
    return ' '.join(x[2] for x in top_n)  # Соединяем предложения




