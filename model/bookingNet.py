# 整个实验过程中使用的主要网络结构
# 是一个多层的MLP结构
import torch
import torch.nn as nn

from model.mlp import MLP


class BookingNet(nn.Module):
    def __init__(self, args, features_len: dict):
        """
        初始化整个BookingNet模型，传入的参数是从命令行传入。
        :param args: args parser args.
        """
        super(BookingNet, self).__init__()

        self.position_net = MLP(
            input_channels=features_len["position"],
            output_channels=args.position_mlp_layers,
            activation=args.activation,
            activation_at_final=True
        )

        self.room_net = MLP(
            input_channels=features_len["room"],
            output_channels=args.room_mlp_layers,
            activation=args.activation,
            activation_at_final=True
        )

        self.addition_net = MLP(
            input_channels=features_len["addition"],
            output_channels=args.addition_mlp_layers,
            activation=args.activation,
            activation_at_final=True
        )

        self.review_net = MLP(
            input_channels=features_len["review"],
            output_channels=args.review_mlp_layers,
            activation=args.activation,
            activation_at_final=True
        )

        self.final_net = MLP(
            input_channels=args.position_mlp_layers[-1] + args.room_mlp_layers[-1] + args.review_mlp_layers[-1] +
            args.addition_mlp_layers[-1] + features_len["instant"],
            output_channels=args.final_mlp_layers,
            activation=args.activation,
            activation_at_final=False
        )

    def forward(self, x):
        position_features = self.position_net(x["position"])
        room_features = self.room_net(x["room"])
        addition_features = self.addition_net(x["addition"])
        review_features = self.review_net(x["review"])
        instant_features = x["instant"]
        features = torch.cat(
            [position_features, room_features, addition_features, review_features, instant_features],
            dim=1
        )
        features = self.final_net(features)
        return features
