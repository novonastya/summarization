# Алгоритм TextRank с длиной наибольшей общей подпоследовательности в качестве меры схожести
from itertools import combinations
from Preprocessing import sent_tokenizer
import networkx as nx
from ROUGE_stem_sw import _split_into_words, _len_lcs


def similarity(s1, s2):
    """
    :param s1: Предложение-строка s1
    :param s2: Предложение-строка s2
    :return float: Степень схожести предложений
    """
    l1 = _split_into_words(s1)
    l2 = _split_into_words(s2)
    return _len_lcs(l1, l2)


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
    pairs = combinations(range(len(sentences)), 2)
    scores = [(i, j, similarity(sentences[i], sentences[j])) for i, j in pairs]
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



