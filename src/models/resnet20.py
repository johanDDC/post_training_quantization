import torch
import torch.nn as nn
import torch.nn.init as init


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3), stride=1, padding=1, downsample=False,
                 quantize=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(output_channels)
        )
        self.downsample_block = nn.Identity() if not downsample else nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(output_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.addition = torch.ao.nn.quantized.modules.functional_modules.FloatFunctional()

    def forward(self, x):
        residual = self.downsample_block(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.addition.add(out, residual)
        return self.relu(out)

    def fuse_model(self):
        torch.ao.quantization.fuse_modules(self.block1, [['0', '1', '2']], inplace=True)
        torch.ao.quantization.fuse_modules(self.block2, [['0', '1']], inplace=True)
        if not isinstance(self.downsample_block, nn.Identity):
            torch.ao.quantization.fuse_modules(self.downsample_block, [['0', '1']], inplace=True)


class ResNet20(nn.Module):
    def __init__(self, configuration, num_classes, start_channels=16, quantize=False):
        """
        params:
        =======
        :param configuration: number of resudial blocks before downsampling blocks. Use (3, 2, 2) for default ResNet20
        architecture.
        :param start_channels: number of channels after stem cell.
        """
        super().__init__()
        self.__channels = start_channels
        self.stem = nn.Conv2d(in_channels=3, out_channels=self.__channels, kernel_size=(3, 3), padding=1)
        self.layers = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        for layer_depth, num_blocks in enumerate(configuration):
            self.layers.append(
                nn.Sequential(*[
                    ResidualBlock(
                        input_channels=self.__channels,
                        output_channels=self.__channels,
                        quantize=quantize
                    ) for _ in range(num_blocks)
                ])
            )
            if layer_depth < len(configuration) - 1:
                self.downsample_blocks.append(
                    ResidualBlock(
                        input_channels=self.__channels,
                        output_channels=2 * self.__channels,
                        stride=2,
                        downsample=True,
                        quantize=quantize
                    )
                )
                self.__channels *= 2
            else:
                self.downsample_blocks.append(nn.Identity())

        self.fc = nn.Linear(self.__channels, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        if quantize:
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()
            torch.ao.quantization.get_default_qconfig('x86')
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()

        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        for layer_block, downsample_block in zip(self.layers, self.downsample_blocks):
            x = layer_block(x)
            x = downsample_block(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for layer in self.layers:
            for module in layer:
                module.fuse_model()
        for i in range(len(self.downsample_blocks) - 1):
            self.downsample_blocks[i].fuse_model()
