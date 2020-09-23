import jieba.analyse
from keras.models import load_model
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
start_time = time.time()


jieba.load_userdict('user_dict.txt')
jieba.analyse.set_stop_words('stop_words.txt')
with open('stop_words.txt', 'r', encoding='utf-8') as f:
    stop_words = f.readlines()
stop_words = [stop_word.rstrip() for stop_word in stop_words]
# 用戶輸入對目前汽車的需求概況
input_word = '''

三個結論

1.要外型酷炫 方便把妹

2.要歐洲車

3.但車的容量不能太大 要有供新手停車容錯的空間



1.請問比較推薦買便宜但普普的新車 還是高檔品牌的二手車呢?

2.一般來說 預算要抓 年薪的多少比例比較好??

3.有推薦的品牌及型號嗎?


謝謝各位
'''
print(input_word)
cut_list = list(jieba.cut(input_word, cut_all=False))
cut_list = [w for w in cut_list if w not in stop_words]
seg = [item for item in cut_list if len(item) > 1]
seg = [' '.join(seg)]
print(seg)

# 加载特征
feature_path = 'models_dc0910_3.pkl'
cv = CountVectorizer(vocabulary=pickle.load(open(feature_path, "rb")))
word = cv.transform(seg).toarray()

model = load_model('car_nlp0910_3.h5')
lb = pickle.loads(open('lb_nlp0913_3.pickle', "rb").read())

# classify the input word
result = model.predict(word)[0]

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

for x in largest_indices(result, 3)[0]:
    print(lb.classes_[x])

# print (result.shape)
# proba = np.max(result)
# idx = np.where(result == proba)[0]
# label = lb.classes_[idx]
# label = "{}: {:.2f}%".format(label, proba * 100)
# print(label)

print("--- spend %s seconds ---" % (time.time() - start_time))


