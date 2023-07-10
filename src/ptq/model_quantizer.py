import torch
from torch.utils.data import DataLoader

from data.data import get_train_data
from main import evaluate
from src.models.resnet20 import ResNet20
from src.ptq.activation_observer import SimpleObserver
from src.ptq.wrappers import QuantizedTensor, TempDequantizer


class ModelQuantizer(torch.nn.Module):
    def __init__(self, model, observer_class, num_bits=8, *args, **kwargs):
        super().__init__()
        self.model = model
        self.num_bits = num_bits
        self.calibration_dict = {}
        self.calibration = True
        self.input_observer = observer_class(*args, **kwargs)
        self.output_observer = observer_class(*args, **kwargs)
        self.model.avg_pool = TempDequantizer(observer_class, torch.nn.AdaptiveAvgPool2d((1, 1)),
                                         dtype=torch.int16)

    @torch.no_grad()
    def calibrate(self):
        for idx, param in enumerate(self.model.parameters()):
            self.calibration_dict[idx] = QuantizedTensor(param.data).calibrate(self.num_bits)
            param.requires_grad_(False)

    def quantize(self):
        for idx, param in enumerate(self.model.parameters()):
            param.data = QuantizedTensor.create_instance(param.data, *self.calibration_dict[idx]).quantize()

        self.calibration = False
        input_scale, input_offset = self.input_observer.calculate_qparams()
        input_min, input_max = self.input_observer.quant_min, self.input_observer.quant_max
        self.input_stats = (input_scale, input_offset, input_min, input_max)
        output_scale, output_offset = self.output_observer.calculate_qparams()
        output_min, output_max = self.output_observer.quant_min, self.output_observer.quant_max
        self.output_stats = (output_scale, output_offset, output_min, output_max)

        self.model.avg_pool.quantize()

    def forward(self, x):
        def observed_forward(x):
            x = self.input_observer(x)
            x = self.model(x)
            x = self.output_observer(x)
            return x

        def dequantized_forward(x):
            qtype = self.input_observer.dtype
            x = QuantizedTensor.create_instance(x, *self.input_stats, qtype)
            x = x.quantize()
            x = self.model(x)
            x = QuantizedTensor.create_instance(x, *self.output_stats, qtype)
            x = x.dequantize()
            return x

        if self.calibration:
            return observed_forward(x)
        else:
            return dequantized_forward(x)
