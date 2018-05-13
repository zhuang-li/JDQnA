__author__ = 'zhuangli'
from tfidf import tfidf
from cnn_text_classification import data_helpers
import jieba
import json
JDQnA_path = "../data/jd_train.json"

q_x_text, q_y, q_labels = data_helpers.load_data_and_labels(JDQnA_path, "question")

a_x_text = data_helpers.load_answer_data_and_labels(JDQnA_path)

q_seg = jieba.cut("发错货物", cut_all=True)
q_seg_list = " ".join(q_seg)

q_sims = tfidf.get_tfidf_sim(q_seg_list,q_x_text)

a_sims = tfidf.get_tfidf_sim(q_seg_list,a_x_text)

sims = q_sims + a_sims

sims = sorted(enumerate(sims), key=lambda item: -item[1])

print (a_x_text[sims[0][0]])
print (a_x_text[sims[1][0]])
print (a_x_text[sims[2][0]])
print (a_x_text[sims[3][0]])
print (a_x_text[sims[4][0]])