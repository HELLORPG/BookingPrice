"""
train and val functions.
"""

import json
import os

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


def train(args):
    """
    用于训练。
    :param args:
    :return:
    """
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
    net = BookingNet(args=args, features_len=features_len).to(args.device)
    if args.optimizer == "Adam":
        optimizer = optim.Adam(params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print("Optimizer type %s is not supported." % args.optimizer)
        exit(-1)

    if args.loss_function == "L1":
        loss_function = nn.L1Loss(reduction="mean")
    elif args.loss_function == "SmoothL1":
        loss_function = nn.SmoothL1Loss(reduction="mean")
    elif args.loss_function == "CEL":
        loss_function = nn.CrossEntropyLoss(reduction="mean")
    else:
        print("Loss function type %s is not supported." % args.loss_function)

    # 训练
    for epoch in range(0, args.epoch):
        train_one_epoch(
            epoch=epoch,
            args=args,
            net=net,
            optimizer=optimizer,
            dataloader=train_loader,
            loss_function=loss_function
        )

        if args.with_val == "True":
            evaluate(
                net=net,
                dataloader=val_loader,
                loss_function=loss_function,
                args=args
            )


@torch.no_grad()
def evaluate(args, net, dataloader, loss_function):
    net.eval()
    total_loss = 0.0
    total_len = 0
    correct_list = list()

    for data in dataloader:
        # 将数据放到指定位置
        for k in data.keys():
            data[k] = data[k].to(args.device)

        output = net(data)
        # print(data["price"].shape)
        loss = loss_function(output, data["price"])

        # 计算train的acc list
        preds_labels = preds_to_labels(output, mode="classify")
        correct_list += get_correct_list(preds_labels, data["price"])

        # 记录数据
        batch_len = output.shape[0]
        total_loss += loss.item() * batch_len
        total_len += batch_len

    average_loss = total_loss / total_len
    average_acc = np.sum(np.array(correct_list)) / total_len
    print("val", average_loss, average_acc)

    return {
        "loss": average_loss,
        "acc": average_acc
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
    correct_list = list()

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

        # 计算train的acc list
        pred_lables = preds_to_labels(output, mode="classify")
        correct_list += get_correct_list(pred_lables, data["price"])

        # 记录数据
        batch_len = output.shape[0]
        total_loss += loss.item() * batch_len
        total_len += batch_len

    average_loss = total_loss / total_len
    average_acc = np.sum(np.array(correct_list)) / total_len
    print("train", average_loss, average_acc)

    return {
        "loss": average_loss,
        "acc": average_acc
    }


def preds_to_labels(preds: torch.Tensor, mode="classify"):
    if mode == "classify":
        assert preds.shape[1] == 6
        pred_labels = torch.argmax(preds.cpu(), dim=1)
    else:
        assert preds.shape[1] == 1
    return pred_labels


def get_correct_list(preds: torch.Tensor, targets: torch.Tensor):
    res = list()
    preds = preds.cpu()
    targets = targets.cpu()
    for i in range(0, len(preds)):
        if preds[i] == targets[i]:
            res.append(1)
        else:
            res.append(0)

    return res

