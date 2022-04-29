import argparse

parser = argparse.ArgumentParser()

# 微调参数
parser.add_argument('--gpu_id', default=1, type=int,
                    help="使用的GPU id")
parser.add_argument("--maxlen", default=320, type=int,
                    help="最长句长")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--model_save_path", default='./reranker_model/', type=str,
                    help="微调模型保存路径")
parser.add_argument("--model_timestamp", default='2022-04-23_03_56_41', type=str,
                    help="微调模型保存时间戳")
parser.add_argument("--device", default='cuda', type=str,
                    help="使用GPU")
parser.add_argument("--lstm_dim", default=256, type=int,
                    help="lstm隐藏状态维度")
parser.add_argument("--gru_dim", default=256, type=int,
                    help="gru隐藏状态维度")
parser.add_argument("--do_train", action='store_true', default=False,
                    help="是否微调")
parser.add_argument("--do_predict", action='store_true', default=False,
                    help="是否预测")
parser.add_argument("--do_train_after_pretrain", action='store_true', default=False,
                    help="是否预训练后再微调")
parser.add_argument("--warmup", action='store_true', default=False,
                    help="是否采用warmup学习率策略")
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument("--pre_model_path", default='./pretrain_model/', type=str,
                    help="预训练模型保存路径")
parser.add_argument("--pre_model_timestamp", default='2021-08-09_11_05_24', type=str,
                    help="预训练模型保存时间戳")
parser.add_argument("--lr", default=1e-5, type=float,
                    help="初始学习率")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--epoch", default=5, type=int,
                    help="训练轮次")
parser.add_argument('--seed', type=int, default=42,
                    help="随机种子")
parser.add_argument('--struc', default='cls', type=str,
                    choices=['cls', 'bilstm', 'bigru'],
                    help="下接结构")
parser.add_argument("--dropout_num", default=1, type=int,
                    help="dropout数量")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size")
parser.add_argument("--avg_size", default=16, type=int,
                    help="平均池化窗口尺寸")
parser.add_argument("--use_avg", action='store_true', default=False,
                    help="是否使用平均池化")
parser.add_argument("--use_ema", action='store_true', default=False,
                    help="是否使用指数加权平均")
parser.add_argument('--ema_start_epoch', default=2, type=int,
                    help="第几个epoch使用EMA")
parser.add_argument("--do_adversarial", action='store_true', default=False,
                    help="是否使用对抗训练")
parser.add_argument("--adversarial_type", default='fgm', type=str, choices=['fgm', 'pgd'],
                    help="对抗训练类型")
parser.add_argument("--fgm_epsilon", default=0.2, type=float)
parser.add_argument('--pgd_adv_k', type=int, default=10)
parser.add_argument('--pgd_alpha', type=float, default=0.3)
parser.add_argument('--pgd_epsilon', type=float, default=0.5)
parser.add_argument("--num_classes", default=2, type=int,
                    help="类别数目")
parser.add_argument("--pre_epoch", default=-1, type=int,
                    help="选取哪个epoch的预训练模型")
parser.add_argument("--model_type", default='roberta', type=str,
                    choices=['roberta', 'nezha_wwm', 'nezha_base'],
                    help="预训练模型类型")
parser.add_argument("--train_data_path", default='./data/reranker_train.tsv', type=str)
parser.add_argument("--valid_data_path", default='./data/reranker_valid.tsv', type=str)
parser.add_argument("--test_data_path", default='./data/base_data4rerank.json', type=str)
parser.add_argument("--use_multi_gpu", action='store_true', default=False)
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--max_grad_norm', type=float, default=2)
parser.add_argument("--fp16", action='store_true', default=False,
                    help="是否使用混合精度训练")

args = parser.parse_args()

# 开源预训练模型路径
model_map = dict()

model_map['roberta'] = {
    'model_path': '/home/wangzhili/YangYang/pretrainModel/chinese_roberta_wwm_ext_pytorch/pytorch_model.bin',
    'config_path': '/home/wangzhili/YangYang/pretrainModel/chinese_roberta_wwm_ext_pytorch/config.json',
    'vocab_path': '/home/wangzhili/YangYang/pretrainModel/chinese_roberta_wwm_ext_pytorch/vocab.txt'}
model_map['nezha_wwm'] = {'model_path': '/home/wangzhili/YangYang/pretrainModel/nezha-cn-wwm/pytorch_model.bin',
                          'config_path': '/home/wangzhili/YangYang/pretrainModel/nezha-cn-wwm/config.json',
                          'vocab_path': '/home/wangzhili/YangYang/pretrainModel/nezha-cn-wwm/vocab.txt'}
model_map['nezha_base'] = {'model_path': '/home/wangzhili/YangYang/pretrainModel/nezha-cn-base/pytorch_model.bin',
                           'config_path': '/home/wangzhili/YangYang/pretrainModel/nezha-cn-base/config.json',
                           'vocab_path': '/home/wangzhili/YangYang/pretrainModel/nezha-cn-base/vocab.txt'}

