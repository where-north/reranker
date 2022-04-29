"""
Name : reranker_model.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2021/8/16 20:06
Desc:
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.nn.functional as F
import torch.nn as nn
import torch
from reranker_args import model_map
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from nezha_model_utils.nezha.modeling_nezha import NeZhaModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from nezha_model_utils.nezha.configuration_nezha import NeZhaConfig

MODEL_CONFIG = {'nezha_wwm': 'NeZhaConfig', 'nezha_base': 'NeZhaConfig', 'roberta': 'RobertaConfig'}
MODEL_NAME = {'nezha_wwm': 'NeZhaModel', 'nezha_base': 'NeZhaModel', 'roberta': 'BertModel'}


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        bert_config = globals()[MODEL_CONFIG[args.model_type]].from_json_file(model_map[args.model_type]['config_path'])
        self.bert = globals()[MODEL_NAME[args.model_type]](config=bert_config)
        if not args.use_avg:
            args.avg_size = 1
        self.args = args

        if args.struc == 'cls':
            self.fc = nn.Linear(768 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bilstm':
            self.bilstm = nn.LSTM(768, args.lstm_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.lstm_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bigru':
            self.bigru = nn.GRU(768, args.gru_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.gru_dim * 2 + 1 - args.avg_size, args.num_classes)

        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(args.dropout_num)])

    def get_tokenizer(self):
        return BertTokenizer(
            vocab_file=model_map[self.args.model_type]['vocab_path'],
            do_lower_case=True)

    def forward(self, x):
        output = self.bert(**x)[0]  # 0:sequence_output  1:pooler_output
        if self.args.struc == 'cls':
            output = output[:, 0, :]  # cls

        else:
            if self.args.struc == 'bilstm':
                _, hidden = self.bilstm(output)
                last_hidden = hidden[0].permute(1, 0, 2)
                output = last_hidden.contiguous().view(-1, self.args.lstm_dim * 2)
            elif self.args.struc == 'bigru':
                _, hidden = self.bigru(output)
                last_hidden = hidden.permute(1, 0, 2)
                output = last_hidden.contiguous().view(-1, self.args.gru_dim * 2)

        if self.args.use_avg:
            output = F.avg_pool1d(output.unsqueeze(1), kernel_size=self.args.avg_size, stride=1).squeeze(1)

        if self.args.dropout_num == 1:
            output = self.dropouts[0](output)
            output = self.fc(output)

        return output


class ModelForDynamicLen(nn.Module):
    def __init__(self, bert_config, args):
        super(ModelForDynamicLen, self).__init__()
        MODEL_NAME = {'nezha_wwm': 'NeZhaModel', 'nezha_base': 'NeZhaModel', 'roberta': 'BertModel'}
        self.bert = globals()[MODEL_NAME[args.model_type]](config=bert_config)
        self.args = args

        if args.struc == 'cls':
            self.fc = nn.Linear(768 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bilstm':
            self.bilstm = nn.LSTM(768, args.lstm_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.lstm_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bigru':
            self.bigru = nn.GRU(768, args.gru_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.gru_dim * 2 + 1 - args.avg_size, args.num_classes)

        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(args.dropout_num)])

    def forward(self, input_ids):
        output = None
        if self.args.struc == 'cls':
            output = torch.stack(
                [self.bert(input_id.to(self.args.device))[0][0][0]
                 for input_id in input_ids])

        if self.args.AveragePooling:
            output = F.avg_pool1d(output.unsqueeze(1), kernel_size=self.args.avg_size, stride=1).squeeze(1)

        # output = self.dropout(output)
        if self.args.dropout_num == 1:
            output = self.dropouts[0](output)
            output = self.fc(output)
        else:
            out = None
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(output)
                    out = self.fc(out)
                else:
                    temp_out = dropout(output)
                    out = out + self.fc(temp_out)
            output = out / len(self.dropouts)

        return output
