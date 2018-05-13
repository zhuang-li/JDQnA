import data_helpers
import json

JDQnA_path = "/Users/zhuangli/Documents/JDQnA/data/jd.json"
with open(JDQnA_path) as data_file:
    JDQnA_json = json.load(data_file)


q_list = data_helpers.segment_chinese_text(JDQnA_json,"question")
print (q_list)