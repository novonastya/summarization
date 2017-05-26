# Оценка алгоритмов через подключение к базе данных
import psycopg2
from TextRank import text_rank_extract
from LSA import lsa_extract
from Random import random_extract
from k_means import kmeans_extract
from numpy import array
from Preprocessing import sent_tokenizer
from ROUGE import rouge_n, rouge_l, rouge_s
from ROUGE_stem import rouge_n_s, rouge_l_s, rouge_s_s
from ROUGE_stem_sw import rouge_n_sw, rouge_l_sw, rouge_s_sw
import time

start_time = time.time()
connect = psycopg2.connect(database='news', user='postgres', host='localhost', password='12345')
cursor = connect.cursor()
sum_rouge_1 = array([0.0, 0.0, 0.0])
sum_rouge_2 = array([0.0, 0.0, 0.0])
sum_rouge_l = array([0.0, 0.0, 0.0])
sum_rouge_s = array([0.0, 0.0, 0.0])
sum_rouge_s4 = array([0.0, 0.0, 0.0])
sum_rouge_s9 = array([0.0, 0.0, 0.0])

sum_rouge_1_s = array([0.0, 0.0, 0.0])
sum_rouge_2_s = array([0.0, 0.0, 0.0])
sum_rouge_l_s = array([0.0, 0.0, 0.0])
sum_rouge_s_s = array([0.0, 0.0, 0.0])
sum_rouge_s4_s = array([0.0, 0.0, 0.0])
sum_rouge_s9_s = array([0.0, 0.0, 0.0])

sum_rouge_1_sw = array([0.0, 0.0, 0.0])
sum_rouge_2_sw = array([0.0, 0.0, 0.0])
sum_rouge_l_sw = array([0.0, 0.0, 0.0])
sum_rouge_s_sw = array([0.0, 0.0, 0.0])
sum_rouge_s4_sw = array([0.0, 0.0, 0.0])
sum_rouge_s9_sw = array([0.0, 0.0, 0.0])
count_rows = 0
cursor.execute("SELECT ann_site, full_new FROM news where is_downloads = TRUE and ann_site is not null and ann_site!='' and full_new is not null and full_new!='';")
for row in cursor:

    eval_sent = sent_tokenizer(row[0])
    n = len(eval_sent)
    eval_text = text_rank_extract(row[1], n)
    sum_rouge_1 += rouge_n(eval_text, row[0], 1)
    sum_rouge_2 += rouge_n(eval_text, row[0], 2)
    sum_rouge_l += rouge_l(eval_text, row[0])
    sum_rouge_s += rouge_s(eval_text, row[0], 10000)
    sum_rouge_s4 += rouge_s(eval_text, row[0], 4)
    sum_rouge_s9 += rouge_s(eval_text, row[0], 9)

    sum_rouge_1_s += rouge_n_s(eval_text, row[0], 1)
    sum_rouge_2_s += rouge_n_s(eval_text, row[0], 2)
    sum_rouge_l_s += rouge_l_s(eval_text, row[0])
    sum_rouge_s_s += rouge_s_s(eval_text, row[0], 10000)
    sum_rouge_s4_s += rouge_s_s(eval_text, row[0], 4)
    sum_rouge_s9_s += rouge_s_s(eval_text, row[0], 9)

    sum_rouge_1_sw += rouge_n_sw(eval_text, row[0], 1)
    sum_rouge_2_sw += rouge_n_sw(eval_text, row[0], 2)
    sum_rouge_l_sw += rouge_l_sw(eval_text, row[0])
    sum_rouge_s_sw += rouge_s_sw(eval_text, row[0], 10000)
    sum_rouge_s4_sw += rouge_s_sw(eval_text, row[0], 4)
    sum_rouge_s9_sw += rouge_s_sw(eval_text, row[0], 9)

    count_rows += 1
    if count_rows % 1000 == 0:
        print(count_rows)
        print(row)
        print("ROUGE-1 ", sum_rouge_1/count_rows)
        print("ROUGE-2 ", sum_rouge_2/count_rows)
        print("ROUGE-L ", sum_rouge_l/count_rows)
        print("ROUGE-S ", sum_rouge_s/count_rows)
        print("ROUGE-S4 ", sum_rouge_s4/count_rows)
        print("ROUGE-S9 ", sum_rouge_s9/count_rows)

        print("ROUGE-1-s ", sum_rouge_1_s/count_rows)
        print("ROUGE-2-s ", sum_rouge_2_s/count_rows)
        print("ROUGE-L-s ", sum_rouge_l_s/count_rows)
        print("ROUGE-S-s ", sum_rouge_s_s/count_rows)
        print("ROUGE-S4-s ", sum_rouge_s4_s/count_rows)
        print("ROUGE-S9-s ", sum_rouge_s9_s/count_rows)

        print("ROUGE-1-sw ", sum_rouge_1_sw/count_rows)
        print("ROUGE-2-sw ", sum_rouge_2_sw/count_rows)
        print("ROUGE-L-sw ", sum_rouge_l_sw/count_rows)
        print("ROUGE-S-sw ", sum_rouge_s_sw/count_rows)
        print("ROUGE-S4-sw ", sum_rouge_s4_sw/count_rows)
        print("ROUGE-S9-sw ", sum_rouge_s9_sw/count_rows)

        print("--- %s seconds ---" % (time.time() - start_time))
connect.close()
print("ROUGE-1 ", sum_rouge_1/count_rows)
print("ROUGE-2 ", sum_rouge_2/count_rows)
print("ROUGE-L ", sum_rouge_l/count_rows)
print("ROUGE-S ", sum_rouge_s/count_rows)
print("ROUGE-S4 ", sum_rouge_s4/count_rows)
print("ROUGE-S9 ", sum_rouge_s9/count_rows)

print("ROUGE-1-s ", sum_rouge_1_s/count_rows)
print("ROUGE-2-s ", sum_rouge_2_s/count_rows)
print("ROUGE-L-s ", sum_rouge_l_s/count_rows)
print("ROUGE-S-s ", sum_rouge_s_s/count_rows)
print("ROUGE-S4-s ", sum_rouge_s4_s/count_rows)
print("ROUGE-S9-s ", sum_rouge_s9_s/count_rows)

print("ROUGE-1-sw ", sum_rouge_1_sw/count_rows)
print("ROUGE-2-sw ", sum_rouge_2_sw/count_rows)
print("ROUGE-L-sw ", sum_rouge_l_sw/count_rows)
print("ROUGE-S-sw ", sum_rouge_s_sw/count_rows)
print("ROUGE-S4-sw ", sum_rouge_s4_sw/count_rows)
print("ROUGE-S9-sw ", sum_rouge_s9_sw/count_rows)
print("--- %s seconds ---" % (time.time() - start_time))