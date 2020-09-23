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
請推薦二手的SUV車

因為自己喜歡戶外活動，有在跟家人朋友登山、露營
所以想找一台可以容納大裝備的車子
考量後車廂空間、後座的位置大小跟油耗，以及要有天窗（老婆大人要求！）
目前有在考慮CR-V、Forester
但怕自己考量的不夠週延
想再問大家對於這兩款的想法，或是有沒有其他推薦的車子

補上預算，目前想法是五十萬上下

以上，謝謝大家
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

with open('clf_svc.pickle', 'rb') as f:
    classifier = pickle.load(f)
result = classifier.predict(word)
lb = pickle.loads(open('clf_svc_label.pickle', "rb").read())
label = lb.inverse_transform(result)
print(label[0])

