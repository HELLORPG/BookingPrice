import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_channels: int, output_channels: list, activation="relu",
                 activation_at_final=False, bias=True):
        super(MLP, self).__init__()

        assert len(output_channels) >= 1

        # 初始化线性网络层
        self.fc_layers = nn.ModuleList(
            [nn.Linear(in_features=input_channels, out_features=output_channels[0], bias=bias)] +
            [nn.Linear(in_features=output_channels[i], out_features=output_channels[i+1], bias=bias)
             for i in range(0, len(output_channels)-1)]
        )

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            print("Activation function %s is not supported." % activation)
            exit(-1)

        self.activation_at_final = activation_at_final

        return

    def forward(self, x):
        for i in range(0, len(self.fc_layers)):
            x = self.fc_layers[i](x)
            if self.activation_at_final and i == len(self.fc_layers):   # 此时不需要增加激活函数
                continue
            else:
                x = self.activation(x)
        return x


if __name__ == '__main__':
    mlp_test = MLP(20, [30])
    print(len(mlp_test.fc_layers))
