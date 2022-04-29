"""
Name : post_utils.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/4/23 8:19
Desc:
"""
import json
from collections import defaultdict
from tqdm import tqdm


def split_data():
    """
    把数据切成4份
    """
    file_path = './data/data4rerank.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f'data len: {len(data)}')
    sub_size = int(len(data) / 4)
    for i in range(4):
        start_id = i * sub_size
        end_id = start_id + sub_size
        sub_data = data[start_id:end_id]
        with open(f'./data/data4rerank_{i}', 'w') as w:
            json.dump(sub_data, w)


def rerank_test(only_rerank_pos=False, model_path=''):
    main_path = f'./reranker_model/{model_path}/'
    scores = []
    for i in range(4):
        scores.extend(json.load(open(main_path + f'scores_{i}.json', 'r')))

    """加载top50数据
    单条格式：
    {'q_text': '',
   'q_id': '',
   'top_50': [(doc_id, doc_text), (...)]}
    """
    with open('./data/data4rerank.json', 'r') as f:
        top50_data = json.load(f)
    sort_res = defaultdict(list)
    for line, score in zip(top50_data, scores):
        q_id = line['q_id']
        top_50 = line['top_50']
        if only_rerank_pos:
            # 只对预测为1的段落进行排序
            pos, neg = [], []
            for i in range(len(top_50)):
                if score[i][1] == 1:
                    top_50[i][1] = score[i][0]
                    pos.append(top_50[i])
                else:
                    neg.append(top_50[i])
            sort_top_50_pos = sorted(pos, key=lambda x: x[1], reverse=True)
            sort_top_50_pos_id = [i[0] for i in sort_top_50_pos]
            neg_id = [i[0] for i in neg]
            sort_res[q_id] = sort_top_50_pos_id + neg_id
        else:
            # 对所有段落进行排序
            for i in range(len(top_50)):
                top_50[i][1] = score[i][0]
            sort_top_50 = sorted(top_50, key=lambda x: x[1], reverse=True)
            sort_top_50_id = [i[0] for i in sort_top_50]
            sort_res[q_id] = sort_top_50_id

    with open('./sort_res.json', 'w', encoding='utf-8') as f:
        json.dump(sort_res, f)


def rerank_valid(only_rerank_pos=False, model_path=''):
    main_path = f'./reranker_model/{model_path}/'

    scores = json.load(open(main_path + f'val_scores.json', 'r'))

    """加载top50数据
    单条格式：
    {'q_text': '',
   'q_id': '',
   'top_50': [(doc_id, doc_text), (...)]}
    """
    with open('./data/val_recall_data4rerank.json', 'r') as f:
        top50_data = json.load(f)
    sort_res = defaultdict(list)
    for line, score in zip(top50_data, scores):
        q_id = line['q_id']
        top_50 = line['top_50']
        if only_rerank_pos:
            # 只对预测为1的段落进行排序
            pos, neg = [], []
            for i in range(len(top_50)):
                if score[i][1] == 1:
                    top_50[i][1] = score[i][0]
                    pos.append(top_50[i])
                else:
                    neg.append(top_50[i])
            sort_top_50_pos = sorted(pos, key=lambda x: x[1], reverse=True)
            sort_top_50_pos_id = [i[0] for i in sort_top_50_pos]
            neg_id = [i[0] for i in neg]
            sort_res[q_id] = sort_top_50_pos_id + neg_id
        else:
            # 对所有段落进行排序
            for i in range(len(top_50)):
                top_50[i][1] = score[i][0]
            sort_top_50 = sorted(top_50, key=lambda x: x[1], reverse=True)
            sort_top_50_id = [i[0] for i in sort_top_50]
            sort_res[q_id] = sort_top_50_id

    calculate_dev_mrr(sort_res)


def calculate_dev_mrr(sort_res):
    dev_data_path = './data/dev.json'
    search_dev_res_path = './data/dev_res.json'
    dev_data_file = open(dev_data_path, 'r', encoding='utf-8')
    dev_data = [json.loads(i) for i in dev_data_file.readlines()]
    search_dev_res_data = json.load(open(search_dev_res_path, 'r', encoding='utf-8'))

    assert len(search_dev_res_data) == len(sort_res)

    search_mrr = 0
    rerank_mrr = 0

    for item in tqdm(dev_data):
        qid = item['question_id']
        pos_paragraph_ids = [i['paragraph_id'] for i in item['answer_paragraphs']]
        for idx, paragraph_id in enumerate(search_dev_res_data[qid]):
            if paragraph_id in pos_paragraph_ids:
                search_mrr += (1 / (idx + 1))
                break
        for idx, paragraph_id in enumerate(sort_res[qid]):
            if paragraph_id in pos_paragraph_ids:
                rerank_mrr += (1 / (idx + 1))
                break

    print(f'search mrr: {search_mrr / len(search_dev_res_data)}')
    print(f'rerank mrr: {rerank_mrr / len(sort_res)}')


if __name__ == '__main__':
    # split_data()
    rerank_valid(model_path='2022-04-28_07_23_35')
    # rerank_test(model_path='2022-04-27_15_51_29')
