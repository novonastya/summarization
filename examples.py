# Скрипт, печатающий 200 примеров работы алгоритмо в файл
import psycopg2
from TextRank import text_rank_extract
from LSA import lsa_extract
from k_means import kmeans_extract
from Preprocessing import sent_tokenizer

connect = psycopg2.connect(database='news', user='postgres', host='localhost', password='12345')
cursor = connect.cursor()
count_rows = 0
cursor.execute("SELECT ann_site, full_new FROM news where is_downloads = TRUE and ann_site is not null and ann_site!='' and full_new is not null and full_new!='';")
for row in cursor:
    text_1 = ""
    eval_sent = sent_tokenizer(row[0])
    n = len(eval_sent)

    ref_text_rank = text_rank_extract(row[1], n)
    ref_k_means = kmeans_extract(row[1], n)
    ref_LSA = lsa_extract(row[1], n)

    f = open('examples.txt', 'a', encoding='utf8')
    text = '\n' + 'Исходный текст' + '\n' + row[1] + '\n' + 'Образцовый реферат' + '\n' + row[0] + '\n' + 'text_rank' + '\n' + ref_text_rank + '\n' + 'LSA' + '\n' + ref_LSA + '\n' + 'k_means' + '\n' + ref_k_means + '\n'
    f.write(text)
    f.close()
    count_rows += 1
    if count_rows == 200:
        break

