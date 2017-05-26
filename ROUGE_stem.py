# Вычисление метрики rouge с учетом стемминга
from Preprocessing import word_tokenizer
from numpy import array
from nltk.stem.snowball import RussianStemmer


def _split_into_words(text):  # Разбиение на слова с учетом стемминга
    """
    Возвращает список слов из текста
    :param text: Исходный текст
    :return list: Список слов текста
    """
    full_text_words = []
    full_text_words.extend(RussianStemmer().stem(word) for word in word_tokenizer.tokenize(text.lower()))
    return full_text_words


def _get_ngrams(n, text):
    """
    :param n: Длина n-граммы
    :param text: Текст
    :return set: Множество n-грамм текста
    """
    words = _split_into_words(text)
    ngram_set = set()
    word_count = len(words)
    max_index_ngram_start = word_count - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(words[i:i+n]))
    return ngram_set


def rouge_n_s(evaluated, reference, n=2):
    """
    Вычисление ROUGE-N.
    :param evaluated: Реферат, полученный алгоритмом
    :param reference: Реферат, написанный человеком
    :param n: Длина n-граммы
    :return float tuple: p - точность, r - полнота, f_measure - f-мера
    """
    evaluated_ngrams = _get_ngrams(n, evaluated)
    reference_ngrams = _get_ngrams(n, reference)
    evaluated_count = len(evaluated_ngrams)
    reference_count = len(reference_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0 or reference_count == 0:
        p = 0
        r = 0
    else:
        p = overlapping_count / evaluated_count
        r = overlapping_count / reference_count

    if p and r:
        f_measure = 2*r*p/(r+p)
    else:
        f_measure = 0

    return array([p, r, f_measure])


def _get_index_of_lcs(x, y):
    return len(x), len(y)


def _len_lcs(x, y):
    """
    Возвращает длину наибольшей общей подпоследовательности между последовательностями x и y.
    Источник: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    :param x: последовательность слов
    :param y: последовательность слов
    :returns integer: длина LCS между x и y
    """
    table = _lcs(x, y)
    n, m = _get_index_of_lcs(x, y)
    return table[n, m]


def _lcs(x, y):
    """
    Вычисляет длину LCS между двумя строками.
    Источник: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    :param x: последовательность слов
    :param y: последовательность слов
    :returns table: словарь координат и длин lcs
    """
    n, m = _get_index_of_lcs(x, y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i-1] == y[j-1]:
                table[i, j] = table[i-1, j-1] + 1
            else:
                table[i, j] = max(table[i-1, j], table[i, j-1])
    return table


def _recon_lcs(x, y):
    """
    Возвращает LCS между x и y.
    Источник: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    :param x: последовательность слов
    :param y: последовательность слов
    :returns sequence: LCS между x и y
    """
    i, j = _get_index_of_lcs(x, y)
    table = _lcs(x, y)

    def _recon(i, j):
        if i == 0 or j == 0:
            return []
        elif x[i-1] == y[j-1]:
            return _recon(i-1, j-1) + [(x[i-1], i)]
        elif table[i-1, j] > table[i, j-1]:
            return _recon(i-1, j)
        else:
            return _recon(i, j-1)
    recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
    return recon_tuple


def rouge_l_s(evaluated, reference):
    """
    Вычисляет ROUGE_L.
    Источник: ROUGE_paper
    :param evaluated: Реферат, полученный алгоритмом
    :param reference: Реферат, написанный человеком
    :returns float tuple: p - точность, r - полнота, f_measure - f-мера
    """

    reference_words = _split_into_words(reference)
    evaluated_words = _split_into_words(evaluated)
    m = len(reference_words)
    n = len(evaluated_words)
    lcs = _len_lcs(evaluated_words, reference_words)
    if m & n:
        p = lcs/n
        r = lcs/m
    else:
        p = 0
        r = 0
    if p and r:
        f_measure = 2*p*r/(r+p)
    else:
        f_measure = 0
    return array([p, r, f_measure])


def _get_skip_bigrams(text, k):
    """
    Из текста возвращает множество биграмм с пропусками.
    :param text: Исходный текст.
    :param k: Максимальное количество слов между частями биграммы.
    :returns set: Множество биграмм с пропусками
    """

    words = _split_into_words(text)
    skip_bigram_set = set()
    n = len(words)
    m = 0
    for w in words:
        m += 1
        i = m
        while (i-m <= k) & (i < n):
            skip_bigram_set.add((w, words[i]))
            i += 1
    return skip_bigram_set


def rouge_s_s(evaluated, reference, k):
    """
    Вычисляет ROUGE-S.
    Источник: ROUGE_paper.
    :param evaluated: Реферат, полученный алгоритмом.
    :param reference: Реферат, написанный человеком.
    :param k: Максимальное количество слов между частями биграммы.
    :return float tuple: p - точность, r - полнота, f_measure - f-мера
    """
    skip_b_eval = _get_skip_bigrams(evaluated, k)
    skip_b_ref = _get_skip_bigrams(reference, k)

    overlap_skip_b = skip_b_eval.intersection(skip_b_ref)

    overlap_count = len(overlap_skip_b)
    eval_count = len(skip_b_eval)
    ref_count = len(skip_b_ref)
    if eval_count and ref_count:
        p = overlap_count/eval_count
        r = overlap_count/ref_count
    else:
        p = 0
        r = 0
    if p and r:
        f_measure = 2*p*r/(r+p)
    else:
        f_measure = 0
    return array([p, r, f_measure])

