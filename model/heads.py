from torch import nn

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, kernel_size=3, padding=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # 添加卷积层
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,  # 修改为更小的卷积核大小
                    padding=padding
                )
            )

            if i < num_layers - 1:
                self.layers.append(nn.ReLU())    # 替换ReLU
                #self.layers.append(nn.Dropout(0.1))  # 新增Dropout

    def forward(self, x):
        # 输入形状: [B, D, L]
        for layer in self.layers:
            x = layer(x)
        return x  # 输出形状: [B, output_dim, L]