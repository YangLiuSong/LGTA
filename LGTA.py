# encoding: utf-8
'''
@author: yls
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: cugxgyls@gmail.com
@software: pycharm
@file: LGTA.py
@time: 2019/4/10 15:31
@desc:
'''
import numpy as np
import csv,sys
from collections import defaultdict
import random,math
import os


class lgta():
    # 初始化
    def __init__(self,LambdaB,topic_num,region_num):
        # 单词的集合
        self.words = set()
        # 单词w在文档d中出现的频数，c(w,d)
        self.c_w_d = []
        # 文档总数
        self.doc_num = 0
        # 主题数
        self.topic_num = topic_num
        # 区域数
        self.region_num = region_num
        # 参数LambdaB
        self.LambdaB = LambdaB

        # EM算法更新的参数
        self.p_z_r = []
        self.p_w_z = [defaultdict(float) for i in range(topic_num)]
        # p(r|alpha)
        self.p_r_alpha = []
        # 区域r的期望E与方差D
        self.r_E = []
        self.r_D = []
        # EM算法更新的参数 End

        # 总词数    (sum w in V)(sum d in D)c(w,d)
        self.all_count = 0
        # 每篇文档的经纬度信息
        self.l_d = []
        # 隐藏变量p(r|d,Variable)
        self.p_r_dVariable = []
        # 隐藏变量
        self.sigma_wrz = {}
        # p(z|l,Variable)
        self.p_z_lVariable = []


    def load_sinaData(self,data_csv,e_npy,d_npy):
        # 记录每个区域类别的数量
        r_init = np.zeros([self.region_num])
        # 遍历文档数据data_csv
        with open(data_csv,encoding="UTF-8")as f:
            datas = csv.reader(f)
            for data in datas:
                # 记录每篇文档的ld
                self.l_d.append(np.array([float(str(data[0]).split(",")[0][1:]),float(str(data[0]).split(",")[1][:-1])]))
                # 读取每篇文档的初始区域信息
                r_index = int(data[1])
                r_init[r_index] += 1
                # 读取每篇文档的分词信息
                ws = str(data[2]).split(" ")
                # 给 c(w,d) 添加一个新的数组
                self.c_w_d.append(defaultdict(int))
                # 遍历每个文档d中的所有单词
                for word in ws:
                    # 将单词w加入集合W中
                    self.words.add(word)
                    # 记录每个词w在每个文档d中出现的词频，即c（d，w）
                    self.c_w_d[-1][word] += 1
        # 遍历文档数据data_csv End
        # 计算初始的p(r|alpha)
        self.p_r_alpha = [r_init[i]/sum(r_init) for i in range(self.region_num)]
        # 计算初始的p(r|alpha) End

        # 读取区域的初始分布的期望与方差信息
        # 期望
        for e in np.load(e_npy):
            self.r_E.append(e)
        # 方差
        for d in np.load(d_npy):
            self.r_D.append(d)
        # 调用初始化参数函数
        self.init_params()


    def init_params(self):
        # 对区域r的每个主题z的概率p(z|r)赋予初值,使求和 p(z|r) = 1
        for i in range(self.region_num):
            zr = [random.random() for i in range(self.topic_num)]
            zr_sum = sum(zr)
            self.p_z_r.append([zr[i] / zr_sum for i in range(self.topic_num)])

        # 计算总词数
        for c_d in self.c_w_d:
            for c_w in c_d:
                self.all_count += c_d[c_w]

        # 给文档中每个词一个初始化的概率分布
        # 初始化p(w|z)
        for z in self.p_w_z:
            for word in self.words:
                z[word] = random.random()

    # 训练函数
    def train(self,max_iter=50):
        # 获取总的文档数量
        self.doc_num = len(self.c_w_d)
        for epoch in range(max_iter):
            print("epoch:" + str(epoch))
            self.p_r_dVariable = []
            # # # # # E Step
            print("E Step begins starting:")
            # 遍历所有文档,计算p(r|d,Variable)
            print("E Step calculate and update p(r|d,Variable):")
            for i in range(self.doc_num):
                print("Document:{0}".format(i))
                # 计算分子
                numerator = []
                for r_index in range(self.region_num):
                    # 计算p(wd|r,Variable)
                    p_wd_rVariable = 1.0
                    for w in self.c_w_d[i]:
                        # 计算p(w|B)
                        sum_cwd = 0
                        for d_id in range(self.doc_num):
                            sum_cwd += self.c_w_d[d_id][w]
                        p_w_B = sum_cwd / self.all_count
                        # 计算p(w|B) End

                        # 计算（sum z in Z）p(w|z)p(z|r)
                        sum_pwz_pzr = sum(
                                [self.p_z_r[r_index][t_id] * self.p_w_z[t_id][w] for t_id in range(self.topic_num)])
                        # 计算（sum z in Z）p(w|z)p(z|r) End

                        # 计算p(w|r,Variable)
                        p_w_rVariable = self.LambdaB * p_w_B + (1.0 - self.LambdaB) * sum_pwz_pzr
                        # 计算p(w|r,Variable) End
                        p_wd_rVariable *= math.pow(p_w_rVariable, self.c_w_d[i][w])  # 连乘计算 p(w|r,Varible)的c(w,d)次方
                    # 计算p(wd|r,Variable) End

                    # 计算p(ld|r,Variable)
                    # np.array 将数组转换成矩阵       np.linalg.inv 求逆矩阵      np.array.transpose   求转置矩阵     np.linalg.det 求矩阵行列式
                    n = -np.mat(self.l_d[i] - self.r_E[r_index]) * np.linalg.inv(np.array(self.r_D[r_index])) * np.mat(self.l_d[i] - self.r_E[r_index]).transpose()
                    p_ld_rVariable = 1.0 / (2.0 * np.pi * np.sqrt(abs(np.linalg.det(np.array(self.r_D[r_index]))))) * np.exp(np.linalg.det(n) / 2.0)
                    # 计算p(ld|r,Variable) End

                    # 计算 p(wd,ld|r,Variable) = p(ld|r,Variable) * p(wd|r,Variable)
                    p_wdld_rVariable = p_wd_rVariable * p_ld_rVariable
                    # 计算 p(wd,ld|r,Variable) = p(ld|r,Variable) * p(wd|r,Variable) End

                    # 计算分子p(r|alpha) * p(wd,ld|r,Variable)
                    numerator.append(self.p_r_alpha[r_index] * p_wdld_rVariable)
                # 计算分子 End

                # 计算分母
                denominator = sum(numerator)
                # 计算分母 End
                # 计算p(r|d,Variable)
                self.p_r_dVariable.append([numerator[r] / denominator for r in range(self.region_num)])
            # 输出每次迭代后的情况
            print(self.p_r_dVariable)
            print("E Step calculate and update p(r|d,Variable) is over!!!")
            # 遍历文档,计算p(r|d,Variable) End


            print("E Step is over!!!")
            # # # # # E Step End

            # M Step
            print("M Step begins starting!!!")
            # 计算p(r|alpha)
            print("M Step calculate and update p(r|alpha):")
            for r_id in range(self.region_num):
                r_alpha_numerator = 0.0
                for prd in self.p_r_dVariable:
                    r_alpha_numerator += prd[r_id]
                self.p_r_alpha[r_id] = r_alpha_numerator / self.doc_num
            print(self.p_r_alpha)
            print("M Step calculate and update p(r|alpha) is over!!!")
            # 计算p(r|alpha) End

            # 计算Er
            print("M Step calculate and update r_E:")
            for r_id in range(self.region_num):
                E_numerator = 0.0
                E_denominator =0.0
                for d_id in range(self.doc_num):
                    E_numerator += self.p_r_dVariable[d_id][r_id] * self.l_d[d_id]
                    E_denominator+= self.p_r_dVariable[d_id][r_id]
                self.r_E[r_id] = E_numerator / E_denominator
            print(self.r_E)
            print("M Step calculate and update r_E is over!!!")
            # 计算Er End

            # 计算Dr
            print("M Step calculate and update r_D:")
            for r_id in range(self.region_num):
                D_numerator = 0.0
                D_denominator = 0.0
                for d_id in range(self.doc_num):
                    D_numerator += (np.mat(self.l_d[d_id] - self.r_E[r_id]).transpose() * np.mat(self.l_d[d_id] - self.r_E[r_id]) * self.p_r_dVariable[d_id][r_id])
                    D_denominator += self.p_r_dVariable[d_id][r_id]
                self.r_D[r_id] = np.array(D_numerator / D_denominator)
            print(self.r_D)
            print("M Step calculate and update r_D is over!!!")
            # 计算Dr End

            # 第二层EM算法起始
            print("Second EM algorithm is starting!!!")
            for se_epoch in range(10):
                # 计算sigma(w,r,z),求解p(z|r)和p(w|z)
                # print("Second-E Step calculate and update sigma_wrz:")
                self.sigma_wrz = {}
                for second_r in range(self.region_num):
                    for second_z in range(self.topic_num):
                        for second_w in self.words:
                            wrz = (second_w, second_r, second_z)
                            se_numerator = (1.0 - self.LambdaB) * self.p_w_z[second_z][second_w] * self.p_z_r[second_r][
                                second_z]
                            # 计算p(w|B)
                            sum_cwd = 0.0
                            for d_id in range(self.doc_num):
                                sum_cwd += self.c_w_d[d_id][second_w]
                            se_p_w_B = sum_cwd / self.all_count
                            # 计算p(w|B) End
                            # 计算（sum z in Z）p(w|z)p(z|r)
                            se_sum_pwz_pzr = sum(
                                [self.p_z_r[r_id][second_z] * self.p_w_z[second_z][second_w] for r_id in
                                 range(self.region_num)])
                            # 计算（sum z in Z）p(w|z)p(z|r) End
                            se_denominator = self.LambdaB * se_p_w_B + (1.0 - self.LambdaB) * se_sum_pwz_pzr
                            self.sigma_wrz[wrz] = se_numerator / se_denominator
                # print(self.sigma_wrz)
                # print("Second-E Step calculate and update sigma_wrz is over!!!")
                # 计算sigma(w,r,z) End

                # 计算p(z|r)
                # print("Second-M Step calculate and update p(z|r):")
                for r in range(self.region_num):
                    pzr_numerator = []
                    for z in range(self.topic_num):
                        _numerator = 0.0
                        for w in self.words:
                            for d_id in range(self.doc_num):
                                _numerator += self.c_w_d[d_id][w] * self.p_r_dVariable[d_id][r] * self.sigma_wrz[(w,r,z)]
                        pzr_numerator.append(_numerator)
                    pzr_denominator = sum(pzr_numerator)
                    for z in range(self.topic_num):
                        self.p_z_r[r][z] = pzr_numerator[z] / pzr_denominator
                # print(self.p_z_r)
                # print("Second-M Step calculate and update p(z|r) is over!!!")
                # 计算p(z|r) End

                # 计算p(w|z)
                # print("Second-M Step calculate and update p(w|z):")
                for z in range(self.topic_num):
                    pwz_numerator = [defaultdict(float) for i in range(len(self.words))]
                    pwz_denominator = 0.0
                    for w in self.words:
                        _numerator = 0.0
                        for d in range(self.doc_num):
                            for r in range(self.region_num):
                                _numerator += self.c_w_d[d][w] * self.p_r_dVariable[d][r] * self.sigma_wrz[(w,r,z)]
                        pwz_numerator[z][w] = _numerator
                        pwz_denominator +=  _numerator
                    for _w in self.words:
                        self.p_w_z[z][_w] = pwz_numerator[z][_w] / pwz_denominator
                # print(self.p_w_z)
                # print("Second-M Step calculate and update p(w|z) is over!!!")
                # 计算p(w|z) End
            # print("Second EM algorithm is over!!!")
            # 第二层EM算法 End
            print("M Step is over!!!")
            # M Step End

            # 计算p(z|l,Variable)
            for d in range(self.doc_num):
                for r in range(self.region_num):
                    # 计算p(l|r,Variable)
                    # np.array 将数组转换成矩阵       np.linalg.inv 求逆矩阵      np.array.transpose   求转置矩阵     np.linalg.det 求矩阵行列式
                    n_ = -np.mat(self.l_d[d] - self.r_E[r]) * np.linalg.pinv(self.r_D[r]) \
                         * np.mat(self.l_d[d] - self.r_E[r]).transpose()
                    p_l_rVariable = 1.0 / (2.0 * np.pi * np.sqrt(abs(np.linalg.det(self.r_D[r])))) \
                                    * np.exp(np.linalg.det(n_) / 2.0)
                    # 计算p(l|r,Variable) End
                    self.p_z_lVariable.append(
                        [p_l_rVariable * self.p_z_r[r][z] * self.p_r_alpha[r] for z in range(self.topic_num)])
            # 强制刷新缓冲区
            sys.stdout.flush()
        self.save_result("model_result")


    # 保存结果
    def save_result(self,path):
        print("Start saving model……")
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + "/p_z_lVariable.npy",self.p_z_lVariable)
        np.save(path + "/p_w_z.npy", self.p_w_z)
        np.save(path + "/p_r_alpha.npy", self.p_r_alpha)
        np.save(path + "/p_z_r.npy", self.p_z_r)
        np.save(path + "/r_E.npy", self.r_E)
        np.save(path + "/r_D.npy", self.r_D)
        np.save(path + "/p_r_dVariable.npy", self.p_r_dVariable)
        print("Model is saved!!!")

    # 计算似然函数的log值
    # def log_likelihood(self):
    #     # 获取总的单词数目
    #     doc_num = len(self.n_d_w)
    #     likelihood = 0
    #     for i in range(doc_num):
    #         # 遍每个文档d的词频记录
    #         for w in self.n_d_w[i]:
    #             p_d_w = 0
    #             for k in range(self.topic_num):
    #                 # p(w|d) = 求和p(z|d) * p(w|z)
    #                 p_d_w += self.p_d_z[i][k] * self.p_z_w[k][w]
    #             # 似然函数L
    #             likelihood += self.n_d_w[i][w] * math.log(p_d_w)
    #     return likelihood

    # 输出每个文档对应的地理主题分类
    def print_document_topic(self):
        # print("The p(z|l,Variable) : ")
        # print(lgta_test.p_z_lVariable)
        for d in range(self.doc_num):
            p = self.p_z_lVariable[d]
            print('Document—{0} Topic is {1}'.format(d,p.index(max(p))))



    # 输出主题的前几关键词
    def print_topic_word(self,num):
        # 循环预定义的主题数量次数
        for k in range(self.topic_num):
        #     # Python2.7的写法
        #     # words = self.p_z_w[k].items()
        #     # words.sort(key=lambda x:x[1],reverse=True)
        #
            # Python3之后的写法
            # 按关键词的概率排序
            words = sorted(self.p_w_z[k].items(), key=lambda x:x[1],reverse=True)
            print('topic:{0}'.format(k))
            # 防止某一主题关键词过少
            if num > len(words):
                num = len(words)
            # 循环输出关键词
            for i in range(num):
                print(words[i])

if __name__ == '__main__':
    lgta_test = lgta(0.9,5,5)
    lgta_test.load_sinaData("../LGTA/test_data.csv","../LGTA/E.npy","../LGTA/D.npy")
    # print(lgta_test.c_w_d)
    # print(lgta_test.p_r_alpha)
    # print(lgta_test.r_E)
    # print(lgta_test.r_D)
    # print(len(lgta_test.words))
    # print(lgta_test.p_w_z)
    # print(len(lgta_test.p_w_z[2]))
    # print(lgta_test.p_z_r)
    # print(len(lgta_test.p_z_r))
    lgta_test.train(2)
    lgta_test.print_topic_word(5)
    lgta_test.print_document_topic()