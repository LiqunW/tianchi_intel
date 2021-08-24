
chars = r'é€íñó￡¢￥áèàäúöüî＄ìÉɪêμãÍòÑôÓıÁ฿✕ʊə¥Üń§œÀłİÅçÈ∧ęŞâāĄşǵïβⅩÇйЙÚ'  # 选出的数量较少的字符

with open(r'/work/dataset/train_data1.txt','r',encoding='utf-8') as f:
    lines = f.read().splitlines()
with open(r'/work/dataset/train_data_min.txt','w',encoding='utf-8') as f2:
    for l in lines:
        try:
            _, label = l.split('\t')
            for c in chars:
                if c in label:
                    f2.write(l + '\n')
                    break
        except:
            continue
