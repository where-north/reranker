# -*- coding: utf-8 -*-
"""
Name : utils.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2021/8/4 20:32
Desc:
"""
import os
import pandas as pd
import logging
import datetime
import random
import torch
import numpy as np
import json


def replace_char(char):
    char = char.replace(',', '，')
    char = char.replace('?', '？')
    char = char.replace('!', '！')

    return char


def load_data(filename):
    """加载数据
    单条格式：`query null para_text label` (`\t` seperated, `null` represents invalid column.)
    """
    data = pd.read_csv(filename, sep='\t', names=['question', 'nan', 'passage', 'label'])
    data = data.drop('nan', axis=1)
    print(f'{filename} 原始数据量：{len(data)}')
    data = data.dropna()
    print(f'{filename} 去除空行后数据量：{len(data)}')
    data['question'] = data['question'].apply(replace_char)
    data['passage'] = data['passage'].apply(replace_char)
    D = []
    if 'label' in data.columns:
        for text1, text2, label in zip(data['question'], data['passage'], data['label']):
            D.append((text1, text2, int(label)))
    else:
        for text1, text2 in zip(data['question'], data['passage']):
            label = -100
            D.append((text1, text2, label))
    return D


def load_test_data(filename):
    """加载测试数据
    单条格式：
    {'q_text': '',
   'q_id': '',
   'top_50': [(doc_id, doc_text), (...)]}
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    D, label = [], -100
    for line in data:
        q_text = line['q_text']
        doc_id_texts = line['top_50']
        for item in doc_id_texts:
            doc_text = item[1]
            D.append((replace_char(q_text), replace_char(doc_text), label))

    return D


def get_save_path(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    args.model_save_path = args.model_save_path + "{}/".format(timestamp)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    argsDict = args.__dict__
    with open(args.model_save_path + 'args.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def get_logger(log_file, display2console=True):
    """
    定义日志方法
    :param display2console:
    :param log_file:
    :return:
    """
    # 创建一个logging的实例 logger
    logger = logging.getLogger(log_file)
    # 设置logger的全局日志级别为DEBUG
    logger.setLevel(logging.DEBUG)
    # 创建一个日志文件的handler，并且设置日志级别为DEBUG
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # 创建一个控制台的handler，并设置日志级别为DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 设置日志格式
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s  - %(levelname)s - %(message)s")
    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add ch and fh to logger
    if display2console:
        logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def truncate_sequences(maxlen, index, *sequences):
    """
    截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences


class ModelSaver:
    def __init__(self, args, patience=10000, metric='max'):
        """
        :param args:
        :param patience: 连续几个epoch无提升，停止训练
        """
        if metric == 'max':
            self.best_score = float('-inf')
        elif metric == 'min':
            self.best_score = float('inf')
        if not os.path.exists(args.model_save_path):
            os.mkdir(args.model_save_path)
        self.args = args
        self.metric = metric
        self.bad_perform_count = 0
        self.patience = patience

    def save(self, eval_score, epoch, logger, model):
        do_save = None
        if self.metric == 'max':
            do_save = eval_score > self.best_score
        elif self.metric == 'min':
            do_save = eval_score < self.best_score
        if do_save:
            self.bad_perform_count = 0
            self.best_score = eval_score
            save_path = self.args.model_save_path + f'/{self.args.model_type}_{self.args.struc}_best_model.pth'
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
        else:
            self.bad_perform_count += 1
            if self.bad_perform_count > self.patience:
                return True
        save_path = self.args.model_save_path + f'/{self.args.model_type}_{self.args.struc}_epoch{epoch + 1}.pth'
        torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
        logger.info(f'save model in {save_path}')
