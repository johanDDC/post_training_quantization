import torch

from torch.ao.quantization.observer import ObserverBase


class SimpleObserver(ObserverBase):
    def __init__(self, dtype=torch.quint8,
                 quant_min=0, quant_max=None, qscheme=None):
        super().__init__(dtype=dtype)
        self.min = torch.tensor(float("inf"))
        self.max = torch.tensor(float("-inf"))
        self.qscheme = torch.per_tensor_affine if qscheme is None else qscheme

        if quant_max is not None:
            self.quant_min = quant_min
            self.quant_max = quant_max
        else:
            if dtype == torch.float16:
                info = torch.finfo(torch.float16)
                self.quant_min = info.min
                self.quant_max = info.max
            elif dtype == torch.int16:
                self.quant_min = -2 ** 15
                self.quant_max = 2 ** 15 - 1
            elif dtype == torch.quint8:
                self.quant_min = 0
                self.quant_max = 2 ** 8 - 1
            elif dtype == torch.qint8:
                self.quant_min = -2 ** 7
                self.quant_max = 2 ** 7 - 1
            elif dtype == torch.quint4x2:
                self.quant_min = 0
                self.quant_max = 2 ** 4 - 1
            elif dtype == torch.quint2x4:
                self.quant_min = 0
                self.quant_max = 2 ** 2 - 1
            else:
                raise NotImplementedError

    def forward(self, x):
        x_min, x_max = torch.aminmax(x)
        self.min = torch.min(x_min, self.min)
        self.max = torch.max(x_max, self.max)
        return x

    def calculate_qparams(self):
        quant_range = self.quant_max - self.quant_min
        offset = torch.zeros_like(self.max, dtype=torch.int)
        if self.qscheme == torch.per_tensor_symmetric:
            absmax = torch.max(torch.abs(self.min), torch.abs(self.max))
            scale_factor = 2 * absmax / quant_range
        else:
            scale_factor = (self.max - self.min) / quant_range
            offset = -torch.round(self.min / scale_factor).to(torch.int) + self.quant_min
            offset = torch.clip(offset, self.quant_min, self.quant_max)
        return scale_factor, offset


class PerChannelObserver(SimpleObserver):
    def __init__(self, dtype=torch.quint8,
                 quant_min=0, quant_max=None, qscheme=None, ch_axis=0):
        qscheme = torch.per_channel_affine if qscheme is None else qscheme
        super().__init__(dtype, quant_min, quant_max, qscheme)
        self.min = None
        self.max = None
        self.ch_axis = ch_axis

    def forward(self, x):
        x_flat = x.flatten(1)
        if self.min is None or self.max is None:
            self.min, self.max = torch.aminmax(x_flat, dim=1)
        else:
            x_min, x_max = torch.aminmax(x_flat, dim=1)
            self.min = torch.min(x_min, self.min)
            self.max = torch.max(x_max, self.max)
        return x

    def calculate_qparams(self):
        quant_range = self.quant_max - self.quant_min
        offset = torch.zeros_like(self.max, dtype=torch.int)
        if self.qscheme == torch.per_channel_symmetric:
            absmax = torch.max(torch.abs(self.min), torch.abs(self.max))
            scale_factor = 2 * absmax / quant_range
        else:
            scale_factor = (self.max - self.min) / quant_range
            offset = -torch.round(self.min / scale_factor).to(torch.int) + self.quant_min
            offset = torch.clip(offset, self.quant_min, self.quant_max)
        return scale_factor, offset
