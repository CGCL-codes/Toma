import javalang
import Levenshtein
import pandas as pd
import time
import multiprocessing as mp
import tqdm
import math
from functools import partial


def get_sim(tool, dataframe):
    inputpath = '/id2sourcecode/'

    sim = []
    for _, pair in dataframe.iterrows():
        id1, id2 = pair.FunID1, pair.FunID2

        sourcefile1 = inputpath + str(id1) + '.java'
        sourcefile2 = inputpath + str(id2) + '.java'
        try:
            similarity = runner(tool, sourcefile1, sourcefile2)
        except Exception as e:
            similarity = repr(e).split('(')[0]
            log = "\n" + time.asctime() + "\t" + tool + "\t" + str(id1) + "\t" + str(id2) + "\t" + similarity
            logfile.writelines(log)
            similarity = 'False'
        print(similarity)
        sim.append(similarity)

    return sim


def getCodeBlock(file_path):     # 只对变量名做了归一化
    block = []
    # print(file_path)
    with open(file_path, 'r') as temp_file:
        lines = temp_file.readlines()
        for line in lines:
            tokens = list(javalang.tokenizer.tokenize(line))
            for token in tokens:
                if type(token) == javalang.tokenizer.Identifier:
                    block.append("id")
                else:
                    block.append(token.value)
    return block


def getCodeBlock_type(file_path):     # 类型
    block = []
    # print(file_path)
    with open(file_path, 'r') as temp_file:
        lines = temp_file.readlines()
        for line in lines:
            tokens = list(javalang.tokenizer.tokenize(line))
            for token in tokens:
                token_type = str(type(token))[:-2].split(".")[-1]
                block.append(token_type)
    return block


def getCodeBlock_token_and_type(file_path):     # 类型加token
    block = []
    # print(file_path)
    with open(file_path, 'r') as temp_file:
        lines = temp_file.readlines()
        for line in lines:
            tokens = list(javalang.tokenizer.tokenize(line))
            for token in tokens:
                if type(token) == javalang.tokenizer.Identifier:
                    block.append("id")
                else:
                    block.append(token.value)
                token_type = str(type(token))[:-2].split(".")[-1]
                block.append(token_type)
    return block


def runner(tool, sourcefile1, sourcefile2):
    block1 = getCodeBlock_type(sourcefile1)
    block2 = getCodeBlock_type(sourcefile2)
    if tool == 't1':
        return Jaccard_sim(block1, block2)
    elif tool == 't2':
        return Dice_sim(block1, block2)
    elif tool == 't3':
        return Jaro_sim(block1, block2)
    elif tool == 't4':
        return Jaro_winkler_sim(block1, block2)
    elif tool == 't5':
        return Levenshtein_sim(block1, block2)
    elif tool == 't6':
        return Levenshtein_ratio(block1, block2)


def intersection_and_union(group1, group2):
    intersection = 0
    union = 0
    triads_num1 = {}
    triads_num2 = {}
    for triad1 in group1:
        triads_num1[triad1] = triads_num1.get(triad1, 0) + 1
    for triad2 in group2:
        triads_num2[triad2] = triads_num2.get(triad2, 0) + 1

    for triad in list(set(group1).union(set(group2))):
        intersection += min(triads_num1.get(triad, 0), triads_num2.get(triad, 0))
        union += max(triads_num1.get(triad, 0), triads_num2.get(triad, 0))
    return intersection, union


def Jaccard_sim(group1, group2):
    # Jaccard 系数

    intersection, union = intersection_and_union(group1, group2)
    # 除零处理
    sim = float(intersection) / union if union != 0 else 0
    return sim


def Dice_sim(group1, group2):
    # Dice 系数
    intersection, union = intersection_and_union(group1, group2)
    # 除零处理
    sim = 2 * float(intersection) / (len(group1) + len(group2)) if (len(group1) + len(group2)) != 0 else 0
    return sim


def Jaro_sim(group1, group2):
    # Jaro相似性

    sim = Levenshtein.jaro(group1, group2)
    return sim


def Jaro_winkler_sim(group1, group2):
    # Jaro_winkler相似性
    sim = Levenshtein.jaro_winkler(group1, group2)
    return sim


def Levenshtein_sim(group1, group2):
    # Levenshtein 距离 编辑距离(EditorDistance)
    distance = Levenshtein.distance(group1, group2)
    return distance


def Levenshtein_ratio(group1, group2):
    # Levenshtein比
    sim = Levenshtein.ratio(group1, group2)
    return sim


def cut_df(df, n):
    df_num = len(df)
    every_epoch_num = math.floor((df_num/n))
    df_split = []
    for index in range(n):
        if index < n-1:
            df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
        else:
            df_tem = df[every_epoch_num * index:]
        df_split.append(df_tem)
    return df_split


def main():
    inputcsv = "/noclone.csv"
    #inputcsv = "/type-1.csv"

    Clonetype = inputcsv.split('/')[-1].split('.')[0]
    if 'noclone' in inputcsv:
        Clonetype = 'noclone'

    #methodtype = 't1'
    pairs = pd.read_csv(inputcsv, header=None)
    pairs = pairs.drop(labels=0)
    pairs.columns = ['FunID1', 'FunID2']

    df_split = cut_df(pairs, 60)

    func1 = partial(get_sim, 't1')
    pool = mp.Pool(processes=60)
    sim_t1 = []
    it_sim_t1 = tqdm.tqdm(pool.imap(func1, df_split))
    for item in it_sim_t1:
        sim_t1 = sim_t1 + item
    pool.close()
    pool.join()

    func2 = partial(get_sim, 't2')
    pool = mp.Pool(processes=60)
    sim_t2 = []
    it_sim_t2 = tqdm.tqdm(pool.imap(func2, df_split))
    for item in it_sim_t2:
        sim_t2 = sim_t2 + item
    pool.close()
    pool.join()

    func3 = partial(get_sim, 't3')
    pool = mp.Pool(processes=60)
    sim_t3 = []
    it_sim_t3 = tqdm.tqdm(pool.imap(func3, df_split))
    for item in it_sim_t3:
        sim_t3 = sim_t3 + item
    pool.close()
    pool.join()

    func4 = partial(get_sim, 't4')
    pool = mp.Pool(processes=60)
    sim_t4 = []
    it_sim_t4 = tqdm.tqdm(pool.imap(func4, df_split))
    for item in it_sim_t4:
        sim_t4 = sim_t4 + item
    pool.close()
    pool.join()

    func5 = partial(get_sim, 't5')
    pool = mp.Pool(processes=60)
    sim_t5 = []
    it_sim_t5 = tqdm.tqdm(pool.imap(func5, df_split))
    for item in it_sim_t5:
        sim_t5 = sim_t5 + item
    pool.close()
    pool.join()

    func6 = partial(get_sim, 't6')
    pool = mp.Pool(processes=60)
    sim_t6 = []
    it_sim_t6 = tqdm.tqdm(pool.imap(func6, df_split))
    for item in it_sim_t6:
        sim_t6 = sim_t6 + item
    pool.close()
    pool.join()
    result = pd.DataFrame({'FunID1': pairs['FunID1'].to_list(), 'FunID2': pairs['FunID2'].to_list(),
                           't1_sim': sim_t1, 't2_sim': sim_t2, 't3_sim': sim_t3,
                           't4_sim': sim_t4, 't5_sim': sim_t5, 't6_sim': sim_t6})
    result.to_csv('./output/' + Clonetype + '_sim.csv', index=False)


if __name__ == '__main__':

    parse_er_file = open('parser_error.txt', 'r')  # 没有则创建
    wrongfile = parse_er_file.read().split(' ')  # 解析失败的文件列表名不包含.java
    logfile = open('errorlog.txt', 'a')
    start = time.time()
    main()

    end = time.time()
    t = end - start
    print(t)
    logfile.close()

