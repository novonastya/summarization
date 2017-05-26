# Алгоритм k-means для суммаризации
from nltk.stem.snowball import RussianStemmer
from Preprocessing import sent_tokenizer, stop_words, word_tokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class StemTokenizer(object):
    def __init__(self):
         self.st = RussianStemmer()

    def __call__(self, doc):
        return [self.st.stem(t) for t in word_tokenizer.tokenize(doc.lower())]


def get_distances_to_clusters(text, n):

    sentences = sent_tokenizer(text)  # Разбиение на предложения
    # Генерация tfidf с учетом стемминга и стоп-слов, термы - униграммы.
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words, tokenizer=StemTokenizer()).fit_transform(sentences)
    km = KMeans(n_clusters=n, random_state=0).fit_transform(tfidf)  # Выполнение k-means
    return km  # Матрица n x k, где n - количество предложений, k - количество кластеров


def get_list_sent(text, n):  # Возвращает список предложений, наиболее близких к каждому из кластеров.
    cluster_matrix = get_distances_to_clusters(text, n)
    id_list = []
    for i in range(n):
        id_list.append(cluster_matrix[:, i].argmin())
    return sorted(id_list)


def kmeans_extract(text, n=5):
    sentences = sent_tokenizer(text)
    count = len(sentences)
    if n <= count:
        l = get_list_sent(text, n)
        return ' '.join(s for i, s in enumerate(sentences) if i in l)
    else:
        return text  # Если заданное количество предложений в будущем реферате больше, чем количество предложений в тексте, то возвращаем сам текст
