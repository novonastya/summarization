# Инициализация всего необходимого для предобработки
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

punkt_param = PunktParameters()
abbreviation = ['тыс', 'руб', 'т.е', 'ул', 'д', 'сек', 'мин', 'т.к', 'т.н', 'т.о', 'ср', 'обл', 'кв', 'пл',
                'напр', 'гл', 'и.о', 'им', 'зам', 'гл', 'т.ч']
punkt_param.abbrev_types = set(abbreviation)  # Для правильного разбиения на предложения
sent_tokenizer = PunktSentenceTokenizer(punkt_param).tokenize
word_tokenizer = RegexpTokenizer(r'\w+')
stop_words = get_stop_words('russian')

