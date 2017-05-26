# Алгоритм TextRank в исходном виде
from itertools import combinations
from nltk.stem.snowball import RussianStemmer
from Preprocessing import sent_tokenizer, word_tokenizer, stop_words
import networkx as nx
import math


def similarity(s1, s2):
    """
    :param s1: Множество слов предложения s1
    :param s2: Множество слов предложения s2
    :return float: Степень схожести предложений
    """
    if not len(s1) or not len(s2):
        return 0.0
    return len(s1.intersection(s2))/(math.log(len(s1)+1) + math.log(len(s2)+1))


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
    words = [set(RussianStemmer().stem(word) for word in word_tokenizer.tokenize(sentence.lower())
                 if word not in stop_words) for sentence in sentences]
    # Список множеств слов с примененным стеммингом, без стоп-слов и приведенных к нижнему регистру.
    # Элемент списка - множество слов одного предложения

    # Нумерация предложений с 0 до len(sent) и генерация всех возможных комбинаций по два:
    pairs = combinations(range(len(sentences)), 2)
    scores = [(i, j, similarity(words[i], words[j])) for i, j in pairs]
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
