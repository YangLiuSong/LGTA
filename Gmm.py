# encoding: utf-8
'''
@author: yls
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: cugxgyls@gmail.com
@software: pycharm
@file: Gmm.py
@time: 2019/4/10 19:50
@desc:
'''
import pymysql
import jieba.analyse
import jieba
import re
from sklearn.mixture import GaussianMixture

db = pymysql.connect("127.0.0.1", "root", "584517880", "beijing")

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def create_LGTAData():
    cur_get = db.cursor(pymysql.cursors.DictCursor)
    cur_insert = db.cursor(pymysql.cursors.DictCursor)
    index = 0
    sql_get = "SELECT text,lat,lon FROM bj_mar LIMIT " + str(index) + ",1000000"
    sql_insert_base = "INSERT INTO lgta (gmm_label,text_words) VALUES (\"{}\",\"{}\")"
    cur_get.execute(sql_get)
    x = []
    t = []
    for c in cur_get:
        lat = float(c["lat"])
        lon = float(c["lon"])
        text = str(c["text"])
        text = re.sub(r"http[^\s]*", " ", text)
        text = re.sub(r"@[^\s]*", " ", text)
        text = re.sub(r"\"", " ", text)
        text = re.sub(r"(?<=\[)[^]]*(?=\])", " ", text)
        text = re.sub(r"(?<=「)[^」]*(?=」)", " ", text)
        text = re.sub('[!"#$%&\'()*+-./:;<=>?@?★、…【】《》？「」“”‘’！[\\]^_`{|}~]', " ", text)
        strinfo = re.compile(' ')
        text = strinfo.sub('', text)
        print(text)
        if text != "":
            sentence_seged = jieba.cut(text)
            stopwords = stopwordslist('stop_words.txt')  # 这里加载停用词的路径
            outstr = ''
            for word in sentence_seged:
                if word not in stopwords:
                    if word != '\t':
                        outstr += word
                        outstr += " "
            print(outstr)
            if outstr:
                x.append([lat, lon])
                t.append(outstr)
        index += 1
    guass = GaussianMixture(n_components=10,random_state=0)
    guass.fit(x)

    labels = guass.predict(x)
    for i in range(len(labels)):
        sql_in = sql_insert_base.format(labels[i],t[i])
        print(sql_in)
        cur_insert.execute(sql_in)
        # db.commit()

def gmm_test():
    cur_get = db.cursor(pymysql.cursors.DictCursor)
    sql_get = "SELECT lat,lon FROM bj_mar LIMIT 100"
    cur_get.execute(sql_get)
    x = []
    for c in cur_get:
        lat = float(c["lat"])
        lon = float(c["lon"])
        x.append([lat, lon])
    guass = GaussianMixture(n_components=5, random_state=0)
    guass.fit(x)

    labels = guass.predict(x)
    # print(labels)

    print(guass.weights_)
    print(guass.means_)
    print(guass.covariances_)
    print(guass.converged_)

# create_LGTAData()
gmm_test()
