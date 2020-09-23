import jieba.analyse
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

jieba.load_userdict('user_dict.txt')
jieba.analyse.set_stop_words('stop_words.txt')
with open('stop_words.txt', 'r', encoding='utf-8') as f:
    stop_words = f.readlines()
stop_words = [stop_word.rstrip() for stop_word in stop_words]
# 用戶輸入對目前汽車的需求概況
input_word = '''
不好意思
        小弟最近想買台SUV來跑山
        就是KUGA,IX-35,CX-5
        個人是覺得KUGA外型我不是很喜歡,但有前向碰撞預警不錯
        IX-35內裝不太好看,但價錢比較便宜
        CX-5外型好看,但之前好像前面擋風玻璃會破的問題也不知道解決了沒
        想請問這三款大家有沒有比較推的?
        謝謝~

'''
print(input_word)
cut_list = list(jieba.cut(input_word, cut_all=False))
cut_list = [w for w in cut_list if w not in stop_words]
seg = [item for item in cut_list if len(item) > 1]
seg = [' '.join(seg)]
print(seg)

# 加载特征
feature_path = 'svm_vc.pkl'
cv = CountVectorizer(vocabulary=pickle.load(open(feature_path, "rb")))
word = cv.transform(seg).toarray()
import xgboost as xgb
classifier = xgb.Booster(model_file='xgboost_model0915.model')
word = xgb.DMatrix(word)

result = int(classifier.predict(word))
lb = pickle.loads(open('clf_svc_label.pickle', "rb").read())
label = lb.classes_[result]
print(label)

