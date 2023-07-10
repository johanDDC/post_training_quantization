import torch


class QuantizedTensor(torch.Tensor):
    def __init__(self, data):
        self.data = data
        self.quantized = False

    @classmethod
    def create_instance(cls, data, scale, offset, min, max, qtype):
        res = cls(data)
        res.scale_factor = scale
        res.offset = offset
        res.min = min
        res.max = max
        res.qtype = qtype
        return res

    def calibrate(self, num_bits: int):
        alpha = self.data.min()
        if num_bits == 16:
            self.qtype = torch.int16
        elif num_bits == 8:
            self.qtype = torch.uint8
        elif num_bits == 4:
            self.qtype = torch.uint8
        elif num_bits == 2:
            self.qtype = torch.uint8
        else:
            raise NotImplementedError()
        if num_bits == 16:
            self.min = -2 ** (num_bits - 1)
            self.max = 2 ** (num_bits - 1) - 1
        else:
            self.min = 0
            self.max = 2 ** num_bits - 1
        self.scale_factor = (self.data.max() - alpha) / (2 ** num_bits - 1)
        self.offset = -torch.round(alpha / self.scale_factor) + self.min
        return self.scale_factor, self.offset, self.min, self.max, self.qtype

    def quantize(self):
        new_data = torch.clip(torch.round(self.data / self.scale_factor + self.offset), self.min, self.max)
        new_data = new_data.to(self.qtype)
        return self.create_instance(new_data, self.scale_factor, self.offset, self.min, self.max, self.qtype)

    def dequantize(self):
        new_data = (torch.dequantize(self.data) - self.offset) * self.scale_factor
        return self.create_instance(new_data, self.scale_factor, self.offset, self.min, self.max, self.qtype)


def as_module(func):
    class AsModule(torch.nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, x):
            return self.func(x)

    return AsModule(func)


class TempDequantizer(torch.nn.Module):
    def __init__(self, observer_class, layer, *args, **kwargs):
        super().__init__()
        self.input_observer = observer_class(*args, **kwargs)
        self.output_observer = observer_class(*args, **kwargs)
        self.layer = layer
        self.calibrate = True
        self.input_stats = None
        self.output_stats = None

    def forward(self, x):
        def observed_forward(x):
            x = self.input_observer(x)
            x = self.layer(x)
            x = self.output_observer(x)
            return x

        def dequantized_forward(x):
            qtype = x.dtype
            x = QuantizedTensor.create_instance(x, *self.input_stats, qtype)
            x = x.dequantize()
            x = self.layer(x)
            x = QuantizedTensor.create_instance(x, *self.output_stats, qtype)
            x = x.quantize()
            return x

        if self.calibrate:
            return observed_forward(x)
        else:
            return dequantized_forward(x)

    def quantize(self):
        self.calibrate = False
        input_scale, input_offset = self.input_observer.calculate_qparams()
        input_min, input_max = self.input_observer.min, self.input_observer.max
        self.input_stats = (input_scale, input_offset, input_min, input_max)
        output_scale, output_offset = self.output_observer.calculate_qparams()
        output_min, output_max = self.output_observer.min, self.output_observer.max
        self.output_stats = (output_scale, output_offset, output_min, output_max)
