"""
train and val functions.
"""

import json
import os
import random

import numpy as np
import torch

from data import InfoFile, BookingToken
from data.dataset import BookingDataset
from data.bookingToken import infos_to_tokens, infos_to_tokens_with_tokenizer, get_tokens_statistic, fix_loss_tokens, \
    norm_tokens, build_tokens_features
from torch.utils.data import DataLoader
from model.bookingNet import BookingNet
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score


def train(args):
    """
    用于训练。
    :param args:
    :return:
    """
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_infos = InfoFile(args.train_path).csv_to_booking_info()
    if args.with_val == "True":
        val_infos = InfoFile(args.val_path).csv_to_booking_info()

    # 进行序列化
    train_tokens, tokenizer = infos_to_tokens(train_infos)
    if args.with_val == "True":
        val_tokens, _ = infos_to_tokens_with_tokenizer(infos=val_infos, tokenizer=tokenizer)

    # 对数据进行处理
    data_statistic = get_tokens_statistic(train_tokens)
    train_tokens = norm_tokens(
        tokens=fix_loss_tokens(tokens=train_tokens, token_statistics=data_statistic),
        token_statistics=data_statistic
    )
    if args.with_val == "True":
        val_tokens = norm_tokens(
            tokens=fix_loss_tokens(tokens=val_tokens, token_statistics=data_statistic),
            token_statistics=data_statistic
        )

    # 构造features
    train_tokens = build_tokens_features(train_tokens)
    if args.with_val == "True":
        val_tokens = build_tokens_features(val_tokens)

    train_dataset = BookingDataset(booking_tokens=train_tokens)
    if args.with_val == "True":
        val_dataset = BookingDataset(booking_tokens=val_tokens)

    # 确定特征维度
    features_len = {
        "position": len(train_tokens[0].features["position"]),
        "room": len(train_tokens[0].features["room"]),
        "addition": len(train_tokens[0].features["addition"]),
        "review": len(train_tokens[0].features["review"]),
        "instant": len(train_tokens[0].features["instant"])
    }

    # 保存data meta
    # 保存tokenizer用于之后的测试
    meta_dir = args.data_meta_dir
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f)
    with open(os.path.join(meta_dir, "statistics.json"), "w") as f:
        json.dump(data_statistic, f)
    with open(os.path.join(meta_dir, "features_len.json"), "w") as f:
        json.dump(features_len, f)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    if args.with_val == "True":
        val_loader = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=args.batch_size
        )

    # 模型和优化器等
    class_weight = [0.0] * 6    # 交叉熵使用的权重列表
    for token in train_tokens:
        class_weight[token.get_price()] += 1.0
    for i in range(0, len(class_weight)):
        class_weight[i] = 1 / class_weight[i]
    class_weight = torch.from_numpy(np.array(class_weight, dtype=np.float32)).to(args.device)
    net = BookingNet(args=args, features_len=features_len).to(args.device)
    # 对模型进行初始化
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    if args.optimizer == "Adam":
        optimizer = optim.Adam(params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print("Optimizer type %s is not supported." % args.optimizer)
        optimizer = None
        exit(-1)

    if args.loss_function == "L1":
        loss_function = nn.L1Loss(reduction="mean")
    elif args.loss_function == "SmoothL1":
        loss_function = nn.SmoothL1Loss(reduction="mean")
    elif args.loss_function == "CEL":
        loss_function = nn.CrossEntropyLoss(reduction="mean", weight=class_weight)
    else:
        print("Loss function type %s is not supported." % args.loss_function)
        loss_function = None
        exit(-1)

    # 创建输出目录
    os.makedirs(args.log_dir, exist_ok=True)

    # 训练
    for epoch in range(0, args.epoch):
        epoch_log = dict()
        epoch_log["epoch"] = epoch

        epoch_log["train"] = train_one_epoch(
                                epoch=epoch,
                                args=args,
                                net=net,
                                optimizer=optimizer,
                                dataloader=train_loader,
                                loss_function=loss_function
                            )
        del epoch_log["train"]["pred_labels"]
        del epoch_log["train"]["true_labels"]

        if args.with_val == "True":
            epoch_log["val"] = evaluate(
                                    net=net,
                                    dataloader=val_loader,
                                    loss_function=loss_function,
                                    args=args
                                )
            del epoch_log["val"]["pred_labels"]
            del epoch_log["val"]["true_labels"]

        with open(os.path.join(args.log_dir, "log.json"), "a") as f:
            f.write(json.dumps(epoch_log))
            f.write("\n")
            # json.dump(epoch_log, f)
        print(epoch_log)

    with open(os.path.join(args.log_dir, "log.json"), "a") as f:
        f.write("==================================================================\n")
    return


@torch.no_grad()
def evaluate(args, net, dataloader, loss_function, is_val=True):
    net.eval()
    total_loss = 0.0
    total_len = 0
    pred_list = list()
    label_list = list()

    for data in dataloader:
        # 将数据放到指定位置
        for k in data.keys():
            data[k] = data[k].to(args.device)

        output = net(data)

        # 记录数据
        batch_len = output.shape[0]
        total_len += batch_len

        # 预测和真值
        pred_list += preds_to_labels(output, mode="classify")
        if is_val:
            loss = loss_function(output, data["price"])
            total_loss += loss.item() * batch_len
            label_list += [label.item() for label in data["price"]]

    if is_val:
        average_loss = total_loss / total_len
        # average_acc = np.sum(np.array(get_correct_list(pred_list, label_list))) / total_len
        # print("val", average_loss, average_acc)
        # print("val", accuracy_score(label_list, pred_list))
        return {
            "loss": average_loss,
            "acc": accuracy_score(label_list, pred_list),
            "pred_labels": pred_list,
            "true_labels": label_list,
            "f1_score": f1_score(label_list, pred_list, average="macro")
        }
    else:
        return {
            "pred_labels": pred_list
        }


def train_one_epoch(epoch: int, args, net, optimizer, dataloader, loss_function):
    """
    训练一个epoch
    :param epoch:
    :param args:
    :param net:
    :param optimizer:
    :param dataloader:
    :param loss_function:
    :return:
    """
    net.train()

    total_loss = 0.0
    total_len = 0
    pred_list = list()
    label_list = list()

    for data in dataloader:
        # 将数据放到指定位置
        for k in data.keys():
            data[k] = data[k].to(args.device)

        optimizer.zero_grad()
        output = net(data)
        # print(data["price"].shape)
        loss = loss_function(output, data["price"])
        loss.backward()
        optimizer.step()

        # 保存pred和真值
        pred_list += preds_to_labels(output, mode="classify")
        label_list += [label.item() for label in data["price"]]

        # 记录数据
        batch_len = output.shape[0]
        total_loss += loss.item() * batch_len
        total_len += batch_len

    average_loss = total_loss / total_len
    # average_acc = np.sum(np.array(get_correct_list(pred_list, label_list))) / total_len
    # print("train", average_loss, average_acc)

    return {
        "loss": average_loss,
        "acc": accuracy_score(label_list, pred_list),
        "pred_labels": pred_list,
        "true_labels": label_list,
        "f1_score": f1_score(label_list, pred_list, average="macro")
    }


def preds_to_labels(preds: torch.Tensor, mode="classify"):
    if mode == "classify":
        assert preds.shape[1] == 6
        pred_labels = torch.argmax(preds.cpu(), dim=1)
    else:
        assert preds.shape[1] == 1
    return pred_labels


def get_correct_list(pred_list: list, label_list: list):
    correct = list()
    for i in range(0, len(pred_list)):
        if pred_list[i] == label_list[i]:
            correct.append(1)
        else:
            correct.append(0)
    return correct

