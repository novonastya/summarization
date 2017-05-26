# Алгоритм, выбирающий предложения для реферата случайным образом
import random
from Preprocessing import sent_tokenizer


def random_extract(text, n):
    sentences = sent_tokenizer(text)
    ratings = list(range(len(sentences)))
    random.shuffle(ratings)
    rand_sent = sorted((r, s) for r, s in zip(ratings, sentences))
    sent_n = rand_sent[:n]
    return ' '.join(s[1] for s in sent_n)


