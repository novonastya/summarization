# Алгоритм LSA для суммаризации
from nltk.stem.snowball import RussianStemmer
import numpy
from numpy.linalg import svd
from Preprocessing import sent_tokenizer, word_tokenizer, stop_words
import math


def create_dictionary(text):
    """
    :param text: Исходный текст
    :return dict: Словарь, в котором ключ - слово, значение - номер слова для нумерации строки матрицы
    """
    words = set(RussianStemmer().stem(word) for word in word_tokenizer.tokenize(text.lower()) if word not in stop_words)
    return dict((w, i) for i, w in enumerate(words))


def create_matrix(text, dictionary):
    """
    Создает матрицу размерности |уникальные слова|х|предложения|,
    где элемент (ij) - количество появлений слова i в предложении j.
    :param text: Исходный текст
    :param dictionary: Словарь (слово, номер слова)
    :return matrix: Матрица размерности |уникальные слова|х|предложения|,
                    где элемент (ij) = 0, если слова i нет в предложении j,
                                     = частота слова i в тексте.
    """
    sentences = sent_tokenizer(text)

    words_count = len(dictionary)
    sentences_count = len(sentences)

    matrix = numpy.zeros((words_count, sentences_count))
    for col, sentence in enumerate(sentences):
        for word in word_tokenizer.tokenize(sentence.lower()):
            word = RussianStemmer().stem(word)
            if word in dictionary:  # Стоп-слова не учитываются в матрице
                row = dictionary[word]
                matrix[row, col] += 1
    rows, cols = matrix.shape
    if rows and cols:
        word_count = numpy.sum(matrix)  # Количество слов в тексте
        for row in range(rows):
            unique_word_count = numpy.sum(matrix[row, :])  # Количество определенного слова в тексте
            for col in range(cols):
                if matrix[row, col]:
                    matrix[row, col] = unique_word_count/word_count
    else:
        matrix = numpy.zeros((1, 1))
    return matrix


def compute_ranks(matrix, n):
    """
    :param matrix: Матрица терм-предложение с TF в элементах.
    :param n: Количество предложений в реферате.
    :return list: Список рангов предложений.
    """
    u_m, sigma, v_m = svd(matrix, full_matrices=False)
    powered_sigma = tuple(s**2 if i < n else 0.0 for i, s in enumerate(sigma))
    ranks = []
    # Итерации по столбцам матрицы (т.е. строкам траспонированной)
    for column_vector in v_m.T:
        rank = sum(s*v**2 for s, v in zip(powered_sigma, column_vector))  # По формуле из статьи lsa2
        ranks.append(math.sqrt(rank))
    return ranks


def lsa_extract(text, n=5):
    """
    Создает реферат методом LSA.
    :param text: Исходный текст
    :param n: Количество предложений в реферате
    :return string: Реферат
    """
    d = create_dictionary(text)
    m = create_matrix(text, d)
    r = compute_ranks(m, n)
    sentences = sent_tokenizer(text)
    rank_sort = sorted(((i, r[i], s) for i, s in enumerate(sentences)), key=lambda x: r[x[0]], reverse=True)
    top_n = sorted(rank_sort[:n])
    return ' '.join(x[2] for x in top_n)
