__author__ = 'zhuangli'
import tensorflow as tf
import numpy as np
import os
from tfidf import tfidf
import jieba
import time
import datetime
from cnn_text_classification import data_helpers
from cnn_text_classification import text_cnn
from tensorflow.contrib import learn
import csv
import json
import sys

tf.flags.DEFINE_string("query", "什么是满返满赠？", "query for the JD QnA task")
FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
def cnn_evaluate(check_point):
    x_raw_title = []
    x_raw_title.append(FLAGS.query)
    print (FLAGS.query)
    vocab_path = os.path.join(check_point, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_title = np.array(list(vocab_processor.transform(x_raw_title)))
    # Evaluation
    # ============================================================================
    checkpoint_file = tf.train.latest_checkpoint(check_point)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            #scores = graph.get_operation_by_name("output/scores").outputs[0]
            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_title), 1, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions,{input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    return all_predictions[0]

def bonus_score_given(label_json_path, train_json_path, bonus_score, winner_id, field):
    with open(label_json_path) as label_data_file:
        label_json = json.load(label_data_file)

    with open(train_json_path) as train_data_file:
        train_json = json.load(train_data_file)
    train_list = []
    for item in train_json:
        q = item["question"]
        if q:
            train_list.append(item)

    train_len = len(train_list)

    bonus_v = np.zeros(train_len)

    winner_name = ""
    for name, id in label_json.items():
        if id == winner_id:
            winner_name = name
    print (winner_name)
    idx = 0

    for item in train_list:
        if item[field] == winner_name:
            bonus_v[idx] = bonus_score
        idx+=1

    return bonus_v

title_label_json = "../data/title.label.json"

list_item_label_json = "../data/list_item.label.json"

train_json_path = "../data/jd_train.json"

title_point = "../cnn_text_classification/runs/1526220923/checkpoints/"
title_id = cnn_evaluate(title_point)
print (title_id)
list_item_point = "../cnn_text_classification/runs/1526224685/checkpoints/"
list_item_id = cnn_evaluate(list_item_point)
#print (list_item_id)

np_title_v = bonus_score_given(title_label_json,train_json_path, 0.15,title_id,"title" )

#print (np_title_v)

np_list_item_v = bonus_score_given(list_item_label_json,train_json_path, 0.1,list_item_id,"list_item")
#print (np_list_item_v)

q_x_text, q_y, q_labels = data_helpers.load_data_and_labels(train_json_path, "question")

a_x_text = data_helpers.load_answer_data_and_labels(train_json_path)

q_seg = jieba.cut(FLAGS.query, cut_all=True)
q_seg_list = " ".join(q_seg)

q_sims = tfidf.get_tfidf_sim(q_seg_list,q_x_text)

a_sims = tfidf.get_tfidf_sim(q_seg_list,a_x_text)

sims = q_sims + a_sims + np_title_v + np_list_item_v

sims = sorted(enumerate(sims), key=lambda item: -item[1])

print (a_x_text[sims[0][0]])
print (a_x_text[sims[1][0]])
print (a_x_text[sims[2][0]])
print (a_x_text[sims[3][0]])
print (a_x_text[sims[4][0]])