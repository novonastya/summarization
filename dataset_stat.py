from numpy import random, mean, var, std
import psycopg2
from Preprocessing import sent_tokenizer, word_tokenizer, stop_words
from nltk.stem.snowball import RussianStemmer


len_sent_text = []
len_sent_ref = []
len_word_text = []
len_word_ref = []
len_word_stem_sw_ref = []
count = 0
connect = psycopg2.connect(database='news', user='postgres', host='localhost', password='12345')
cursor = connect.cursor()
cursor.execute("SELECT ann_site, full_new FROM news where is_downloads = TRUE and ann_site is not null and ann_site!='' and full_new is not null and full_new!='';")
for row in cursor:
    len_sent_ref.append(len(sent_tokenizer(row[0])))
    len_sent_text.append(len(sent_tokenizer(row[1])))
    len_word_ref.append(len(set(word_tokenizer.tokenize(row[0].lower()))))
    len_word_text.append(len(set(word_tokenizer.tokenize(row[1].lower()))))
    len_word_stem_sw_ref.append(len(set(RussianStemmer().stem(word) for word in word_tokenizer.tokenize(row[0].lower())
                                       if word not in stop_words)))
    count += 1
    if count % 1000 == 0:
        print(count)

print("Среднее количесвто предложений в рефератах", mean(len_sent_ref))
print("Среднее количесвто предложений в текстах", mean(len_sent_text))
print("Среднее количесвто слов в рефератах", mean(len_word_ref))
print("Среднее количесвто слов в текстах", mean(len_word_text))
print("Среднее количесвто слов стем и стоп-слова", mean(len_word_stem_sw_ref))

print("Мин количесвто предложений в рефератах", min(len_sent_ref))
print("Мин количесвто предложений в текстах", min(len_sent_text))
print("Мин количесвто слов в рефератах", min(len_word_ref))
print("Мин количесвто слов в текстах", min(len_word_text))
print("Мин количесвто слов стем и стоп-слова", min(len_word_stem_sw_ref))

print("Макс количесвто предложений в рефератах", max(len_sent_ref))
print("Макс количесвто предложений в текстах", max(len_sent_text))
print("Макс количесвто слов в рефератах", max(len_word_ref))
print("Макс количесвто слов в текстах", max(len_word_text))
print("Макс количесвто слов стем и стоп-слова", max(len_word_stem_sw_ref))

print("Среднее отклонение предложений в рефератах", std(len_sent_ref))
print("Среднее количесвто предложений в текстах", std(len_sent_text))
print("Среднее количесвто слов в рефератах", std(len_word_ref))
print("Среднее количесвто слов в текстах", std(len_word_text))
print("Среднее количесвто слов стем и стоп-слова", std(len_word_stem_sw_ref))
