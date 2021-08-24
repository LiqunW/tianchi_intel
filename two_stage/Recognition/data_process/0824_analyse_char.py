import os
from matplotlib import pyplot as plt
import pickle
from collections import Counter
import tqdm


# char_cnt = Counter()
#
# with open(r'/work/dataset/train_data1.txt','r',encoding='utf-8') as f:
#     lines = f.read().splitlines()
#     lines = [i.split('\t')[-1] for i in lines]
#     for l in tqdm.tqdm(lines):
#         for c in l:
#             char_cnt[c]+=1
# res = char_cnt.most_common()
# chars = [i[0] for i in res]
# nums = [i[1] for i in res]
# with open('char_cnt.pkl','wb') as f:
#     pickle.dump([chars, nums], f)
# print()
chars = r'é€íñó￡¢￥áèàäúöüî＄ìÉɪêμãÍòÑôÓıÁ฿✕ʊə¥Üń§œÀłİÅçÈ∧ęŞâāĄşǵïβⅩÇйЙÚ'  # 选出的数量较少的字符

with open('char_cnt.pkl','rb') as f:
    chars, nums = pickle.load(f)
print(chars[83:])
plt.bar(chars, nums)
plt.show()
# print()