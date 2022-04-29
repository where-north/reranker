"""
Name : train_rerank_encoder.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/4/22 15:39
Desc:
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reranker_args import args, model_map
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from reranker_utils.optimizer_tools import *
from reranker_utils.utils import *
from reranker_utils.loss_tools import *
from reranker_utils.adversarial_tools import *
from reranker_utils.Datasets import FineTuneDataset
from reranker_model import Model
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler


# 是否使用多GPU
if args.use_multi_gpu:
    torch.distributed.init_process_group(backend="nccl")
    if args.do_train and args.local_rank == 0:
        # 创建模型保存路径以及日志
        get_save_path(args)
        logger = get_logger(args.model_save_path + '/finetune.log')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # 创建模型保存路径以及日志
    if args.do_train:
        get_save_path(args)
        logger = get_logger(args.model_save_path + '/finetune.log')


def input_to_device(numpy_array, args):
    if args.use_multi_gpu:
        return numpy_array.cuda(args.local_rank, non_blocking=True).long()
    else:
        return numpy_array.to(args.device).long()


def train(model):
    # 读取训练数据
    train_data = load_data(args.train_data_path)
    valid_data = load_data(args.valid_data_path)[:10000]

    if args.use_multi_gpu:
        train_dataset = FineTuneDataset(train_data, args.maxlen, model.get_tokenizer())
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        valid_dataset = FineTuneDataset(valid_data, args.maxlen, model.get_tokenizer())
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    else:
        train_dataset = FineTuneDataset(train_data, args.maxlen, model.get_tokenizer())
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataset = FineTuneDataset(valid_data, args.maxlen, model.get_tokenizer())
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    criterion = nn.BCEWithLogitsLoss()
    saver = ModelSaver(args, patience=2, metric='max')
    optimizer, scheduler = build_optimizer(args, model, total_steps=len(train_loader) * args.epoch)

    if args.use_multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)

    if args.do_adversarial:
        adversary = build_adversary(args, model)
    model.zero_grad()

    scaler = None
    if args.fp16: scaler = GradScaler()

    for epoch in range(args.epoch):
        if args.use_multi_gpu:
            train_sampler.set_epoch(epoch)
        if args.use_ema and epoch + 1 >= args.ema_start_epoch:
            ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.999)
        pbar = tqdm(train_loader, ncols=150)
        losses, acc_list = [], []
        for data in pbar:
            model.train()
            optimizer.zero_grad()

            inputs = {
                'input_ids': input_to_device(data['input_ids'], args),
                'attention_mask': input_to_device(data['attention_mask'], args),
                'token_type_ids': input_to_device(data['token_type_ids'], args),
            }
            data['label'] = input_to_device(data['label'], args).float()

            if args.fp16:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, data['label'])
                scaler.scale(loss).backward()
                if args.do_adversarial:
                    backward_adversarial_loss(args, model, inputs, data['label'], adversary, scaler, criterion)
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, data['label'])
                loss.backward()

                if args.do_adversarial:
                    backward_adversarial_loss(args, model, inputs, data['label'], adversary, scaler, criterion)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()

            if args.warmup: scheduler.step()
            if args.use_ema and epoch + 1 >= args.ema_start_epoch:
                ema.update()

            losses.append(loss.cpu().detach().numpy())
            output_array = outputs.cpu().detach().numpy()
            label_array = data['label'].cpu().detach().numpy()
            acc_list.extend(np.argmax(output_array, axis=1) == np.argmax(label_array, axis=1))
            pbar.set_description(
                f'epoch:{epoch + 1}/{args.epoch} lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.7f} loss:{np.mean(losses):.4f} acc:{(np.sum(acc_list) / len(acc_list)):.3f}')

        if args.use_ema and epoch + 1 >= args.ema_start_epoch:
            ema.apply_shadow()

        if args.local_rank in [0, -1]:
            auc_score, accuracy, report = evaluate(model, valid_loader)
            early_stop = saver.save(auc_score, epoch, logger, model.module if args.local_rank != -1 else model)
            if early_stop:
                break
            logger.info(f'epoch:{epoch + 1}/{args.epoch}, valid auc_score: {auc_score}')
            logger.info(f'epoch:{epoch + 1}/{args.epoch}, vaild accuracy: {accuracy}')
            logger.info(f'{report}')


def evaluate(model, data_loader):
    model.eval()
    true, positive_logits, preds = [], [], []
    pbar = tqdm(data_loader, ncols=150)
    with torch.no_grad():
        for data in pbar:
            data['label'] = data['label'].float()
            inputs = {
                'input_ids': input_to_device(data['input_ids'], args),
                'attention_mask': input_to_device(data['attention_mask'], args),
                'token_type_ids': input_to_device(data['token_type_ids'], args),
            }
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            positive_logit = outputs[:, 1] / (outputs.sum(axis=1) + 1e-8)
            pred = np.argmax(outputs.cpu().numpy(), axis=-1)
            true.extend(np.argmax(data['label'], axis=1))
            positive_logits.extend(positive_logit.cpu().numpy())
            preds.extend(pred)

    auc_score = roc_auc_score(true, positive_logits)
    accuracy = accuracy_score(true, preds)
    report = classification_report(true, preds)

    return auc_score, accuracy, report


def predict(model):
    time_start = time()
    set_seed(args.seed)

    test_data = load_test_data(args.test_data_path)

    test_dataset = FineTuneDataset(test_data, args.maxlen, model.get_tokenizer())
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    model.eval()
    positive_logits, preds = [], []
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=150):
            inputs = {
                'input_ids': input_to_device(data['input_ids'], args),
                'attention_mask': input_to_device(data['attention_mask'], args),
                'token_type_ids': input_to_device(data['token_type_ids'], args),
            }
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            preds.extend(np.float_(np.argmax(outputs.cpu().numpy(), axis=1)))
            positive_logit = outputs[:, 1] / (outputs.sum(axis=1) + 1e-8)
            positive_logits.extend(np.float_(positive_logit.cpu().numpy()))

    print(f'predict data len: {len(positive_logits)}')
    res, temp = [], []
    for score, label in zip(positive_logits, preds):
        temp.append((score, label))
        if len(temp) == 50:
            res.append(temp)
            temp = []

    data_id = args.test_data_path.split('_')[1]
    if len(data_id) == 1:
        out_file = args.model_save_path + args.model_timestamp + f"/scores_{data_id}.json"
    else:
        out_file = args.model_save_path + args.model_timestamp + f"/val_scores.json"
    with open(out_file, 'w') as f:
        json.dump(res, f)

    time_end = time()
    print(f'finish {time_end - time_start}s')


def main():
    set_seed(args.seed)

    model = Model(args=args)

    if args.do_train:
        state_dict = torch.load(model_map[args.model_type]['model_path'])
        model.load_state_dict(state_dict, strict=False)

        if args.use_multi_gpu:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
        else:
            model.to(args.device)

        train(model)
    elif args.do_train_after_pretrain:
        file_dir = args.pre_model_path + args.pre_model_timestamp
        file_list = os.listdir(file_dir)
        for name in file_list:
            if name == f'{args.model_type}.pth' or name.split('.')[-1] != 'pth':
                continue
            model_path = os.path.join(file_dir, name)
            if os.path.isfile(model_path) and name.split('-')[1] == f'epoch{args.pre_epoch}.pth':
                print('pretrain model: ', name)
                state_dict = torch.load(model_path, map_location='cuda')
                model.load_state_dict(state_dict, strict=False)
                model = model.to(args.device)
                train(model)
    elif args.do_predict:
        model.load_state_dict(
            torch.load(args.model_save_path + args.model_timestamp + f'/{args.model_type}_{args.struc}_best_model.pth',
                       map_location='cuda'))
        model = model.to(args.device)
        predict(model)


if __name__ == '__main__':
    main()
