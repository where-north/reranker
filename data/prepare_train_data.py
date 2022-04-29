"""
Name : prepare_train_data.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/4/26 21:56
Desc:
"""
import json
import random
from tqdm import tqdm
import pandas as pd

random.seed(0)

'''
单条训练数据格式：`query null para_text label` (`\t` seperated, `null` represents invalid column.)
为每条问题随机采样4条负样本。
'''

with open('./train_data_top50.json', 'r', encoding='utf-8') as f:
    # 数据格式：{'q_text': '问题', 'q_id': '问题ID', 'top_50': [['段落ID', '段落'],...]
    top50_data = json.load(f)

with open('./train.json', 'r', encoding='utf-8') as f:
    # 数据格式：[{'question_id': '', 'question': '', 'answer_paragraphs': [{'paragraph_id': '', 'paragraph_text': ''}, ...]}]
    train_data = [json.loads(line) for line in f.readlines()]

assert len(top50_data) == len(train_data)

query, para_text, label = [], [], []
for i in tqdm(range(len(train_data))):
    if train_data[i]['question_id'] == top50_data[i]['q_id']:
        question = train_data[i]['question']
        pos_para_pools = set([item['paragraph_text'] for item in train_data[i]['answer_paragraphs']])
        top_para_pools = set([item[1] for item in top50_data[i]['top_50']])
        neg_para_pools = list(top_para_pools - pos_para_pools)
        if len(neg_para_pools) < 2:
            print(train_data[i]['question'])
        sample_neg_number = min(len(pos_para_pools) * 4, len(neg_para_pools))
        random.shuffle(neg_para_pools)
        para_text.extend([neg_para_pools[j] for j in range(sample_neg_number)])
        label.extend([0 for _ in range(sample_neg_number)])
        query.extend([question for _ in range(sample_neg_number)])
        for pos in pos_para_pools:
            query.append(question)
            para_text.append(pos)
            label.append(1)

assert len(query) == len(para_text) == len(label)
print(f'label 1 numbers: {sum(label)}, label 0 numbers: {len(label) - sum(label)}')

sample_rerank_data = pd.DataFrame({
    'query': query,
    'null': ['' for _ in range(len(query))],
    'para_text': para_text,
    'label': label
})

sample_rerank_data = sample_rerank_data.sample(frac=1).reset_index(drop=True)
data_len = len(sample_rerank_data)
print(f'all data len: {data_len}')
train_size = int(data_len * 0.8)
train_data = sample_rerank_data.iloc[:train_size].reset_index(drop=True)
valid_data = sample_rerank_data.iloc[train_size:].reset_index(drop=True)
pd.DataFrame(train_data).to_csv('./reranker_train.tsv',
                                sep='\t', index=False, header=None)
pd.DataFrame(valid_data).to_csv('./reranker_valid.tsv',
                                sep='\t', index=False, header=None)
