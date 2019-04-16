# encoding: utf-8
'''
@author: yls
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: cugxgyls@gmail.com
@software: pycharm
@file: create_csv.py
@time: 2019/4/12 9:41
@desc:
'''
import csv
import pymysql
import numpy
import jieba.analyse
import jieba
import re
from sklearn.mixture import GaussianMixture

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

db = pymysql.connect("127.0.0.1", "root", "584517880", "beijing")

def sqlToCsv(db_name):
    cur_get = db.cursor(pymysql.cursors.DictCursor)
    sql_get = "SELECT text,lat,lon FROM " + db_name + " LIMIT 100"
    cur_get.execute(sql_get)
    t = []
    x = []
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
            outstr = ""
            for word in sentence_seged:
                if word not in stopwords:
                    if word != '\t':
                        outstr += word
                        outstr += " "
            print(outstr)
            if outstr != "":
                x.append([lat, lon])
                t.append(outstr[:-1])

    guass = GaussianMixture(n_components=5, random_state=0)
    guass.fit(x)

    labels = guass.predict(x)
    print(labels)
    rows = []
    for i in range(len(labels)):
        rows.append([x[i],labels[i],t[i]])
    with open("test_data.csv","w",newline="",encoding="UTF-8")as f:
        f_csv = csv.writer(f)
        f_csv.writerows(rows)

    e_mean = guass.means_
    numpy.save("E.npy",e_mean)
    d_covariance = guass.covariances_
    numpy.save("D.npy", d_covariance)
    # rows1 = []
    # for j in range(len(e_mean)):
    #     rows1.append([e_mean[j],d_covariance[j]])
    # with open("test_martix.csv","w",encoding="UTF-8")as f1:
    #     f_csv1 = csv.writer(f1)
    #     f_csv1.writerows(rows1)

if __name__ == '__main__':
    sqlToCsv("bj_mar")
    # print(numpy.array("[40.217603231, 116.25661998]"))