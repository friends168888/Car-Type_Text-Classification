import jieba.analyse
from tensorflow.keras.models import load_model
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
大家假日午安
上次問過但最試試了很多對自己需求稍微變更
想在請問一下大家

自己條件
1.因為我自己187,不想買太小的,但因為只有兩個人需求,也不想買大房車
  個人覺得可能就抓車長460-470 左右應該剛好
2.喜歡四門,不愛休旅,掀背車型
3.希望能有ACC,360環景
4.預算大概抓120以內
5.希望舒適度夠

目前看過的
Altis->頭頂空間太小剃除
他在斗->中規中矩,外型喜歡空間也夠,但配備給得比較少
skoda->目前octiva 四門今年似乎都沒了,要等年底或明年初的改款
       有試乘kamiq,操縱性好,但車道置中不太明顯,然後冷氣不太涼
focus->目前應該算最適合我的,配備操縱空間都很完美,但我覺得座椅舒適
       舒適度試感受最不好的,比較硬而且太短腳的支撐性不好
Altima->這台不會買,單純喜歡就試乘,基本上我覺得開起來感受度是最好的
        馬力夠,Nissan的座椅舒適度真的沒話說,每個朋友都說舒適第一名
        但就太大台不合我標準,但這個舒適度會讓我想看一下仙草

接下來還預計去看馬3以及subaru ,以及等接下來改款的仙草還有Octiva
然後就是在想要不要捏上去的IS 跟C300

大家還有推薦什麼?或可以給我點意見的
感謝!     
'''
print(input_word)
cut_list = list(jieba.cut(input_word, cut_all=False))
cut_list = [w for w in cut_list if w not in stop_words]
seg = [item for item in cut_list if len(item) > 1]
seg = [s for s in seg if len(s) > 0]
print(seg)

import kashgari
model = kashgari.utils.load_model("BiLSTM_Model")

# classify the input word
result = model.predict([seg])[0]

print(result)
