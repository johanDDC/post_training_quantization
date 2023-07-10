# Implementation of Post Training Quantization

## Introduction

This repository contains simple implementation of the PTQ based on 
PyTorch API. It also contains an unfinished custom quantization engine,
that was an attempt to implement PTQ from scratch without using Pytorch
Quantization API. 

## ResNet20 implementation
I [implemented](src/models/resnet20.py) ResNet20 architecture that is described in paper 
[Measuring what Really Matters: Optimizing Neural Networks for TinyML](https://www.researchgate.net/publication/351046093_Measuring_what_Really_Matters_Optimizing_Neural_Networks_for_TinyML).

![](https://www.researchgate.net/profile/Zhongnan-Qu/publication/351046093/figure/fig3/AS:1015228695343106@1619060791401/ResNet-20-architecture.png)

During training, I used RandAugment with 2 augmentations and magnitude 14
(taken from paper [RandAugment: Practical automated data augmentation
with a reduced search space](https://arxiv.org/pdf/1909.13719.pdf)). 
And [MixUp](https://arxiv.org/abs/1710.09412). Model was trained for
50 epochs with 10 warmup epochs. All unspecified training parameters
(including optimizer settings and mixup parameter) were taken from
[ConvNeXt](https://arxiv.org/pdf/2201.03545.pdf) paper. To speed up training
I also used Automatic Mixed Precision.

Model achieves **90.89%** accuracy on test CIFAR-10 split. Weights of
trained model are available by the [link](https://drive.google.com/file/d/128V7OaDkI7oOnpb2p8XXxBJYGEddUVPs/view?usp=sharing).
Also, training logs are available at [W&B project](https://wandb.ai/johan_ddc_team/quatization_simple/runs/18g01tsv?workspace=user-johan_ddc).

You may perform training of ResNet20 on CIFAR-10 dataset using my 
setting by specifying the following command:

`python main.py`

Additionally, there are several command-line arguments available 
that can be used to configure your training.

## PyTorch PTQ

I used post training static quantization setting described in [PyTorch
tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html). 
I implemented ResNet-20 in such a way that it wouldn't necessary to make
any modifications when transitioning to quantization. Thus, I just fused layers, 
defined default qconfing, performed calibration on several training
batches and gained the quantized model.

I also translated model to float16 to perform 16-bit quantization. Although, it is
quite simple, it requires GPU for inference.

## Custom PTQ

### Custom PTQ with PyTorch Quantization API

The core of post training static quantization are Observers, which collect
necessary statistics during calibration and translate activations to proper
range when calibration is done. Thus, I [implement custom Quantization 
Observer]() and attach it to each parameter layer using PyTorch Quantization API.
This solution is hardly distinguished from method described in tutorial. The only
difference is now I should provide custom qconfig with custom Observers whick
will observe both activations and parameters.

I have implemented a `SimpleObserver` that incorporates two quantization schemes:

* `per_tensor_affine`. Involves computation of both quantization parameters:
`scale_factor` and `offset` based on collected statistics.
* `per_tensor_symmetric`. Only requires computation of `scale_factor`.

`SimpleObserver` computes quantization parameters as described in paper
[INTEGER QUANTIZATION FOR DEEP LEARNING INFERENCE:
PRINCIPLES AND EMPIRICAL EVALUATION](https://arxiv.org/pdf/2004.09602.pdf).

Example:
```python
model = ResNet20(configuration=(3, 2, 2), num_classes=10, quantize=True)
model.load_state_dict(torch.load("checkpoints/resnet20_final.pth"))
model.eval()
model.qconfig = torch.ao.quantization.qconfig.QConfig(SimpleObserver.with_args(dtype=torch.quint8),
                                                      SimpleObserver.with_args(dtype=torch.qint8,
                                                                               qscheme=torch.per_tensor_symmetric))
model.fuse_model()
model = torch.ao.quantization.prepare(model, inplace=True)
_, _ = evaluate(model, criterion, train_loader, num_batches=num_calibration_batches)
model_q = torch.ao.quantization.convert(model)
```

Additionally, there are `quant_min` and `quant_max` parameters, which allow us control 
borders of quantization range. Thus, is order to quantize model to lower precision we
just might properly set up these parameters. 

One last thing is if we say "quantize model to N-bits", that means, that we quantize both
activations and model weights.

### Custom quantization engine

Although, the previous solution meets the requirements (it collects
statistics of both weights and activations and computes qiantization
parameters), I felt that it was too simple. I know, that the main steps
of PTSQ are the following:

1. Assign Observer to each parameter layer;
2. Compute quantization parameters for model weights (as it doesn't
require calibration);
3. Perform calibration;
4. Perform quantization using computed parameters.

Thus, I [implemented](src/ptq/model_quantizer.py) `ModelQuantizer` that
quantize model weight to `int16` or `int8` types with 16, 8, 4 or 2 bits
precision. This module should have become a custom quantization engine,
however, I met some complications, I couldn't figure out. 

For instance, ResNet uses AvgPool layer that aggregates information from
several feature map channels by applying mean operation. However, as far as
we work with numbers in int8 representation, the notion of the `mean` operation
become kinda unclear. Thus, the only way I found, to overcome this issue is
dequantize the input on AvgPool layer and then again quantize its output.
That solution lead to catastrophic latency during batches processing.

Because of such issues the possible SOTA quantization engine was not developed. 
Although, my `ModelQuantizer` is useless for model inference, it can be useful
for storing model weights, as it operates with torch.int (not torch.qint) 
representation and can freely quantize and dequantize model weights 
to/from proper range.

# Experiments

## Setup

I treat pretrained ResNet20 without any quantization as a baseline. I conduct
experiments on CPU with only one active thread (except for experiments with 16-bit
quantized model). For each experiment (with the same exception) I fuse model and 
measure the amount of time it takes for the model to process the entire test split.
As a result, I provide test loss and test accuracy. I provide the results for 
16-bit quantized model only once, as the only way to perform 
16-bit quantization is to translate model to fp16 precision (as torch
still doesn't support quantization to qint16). Also, for each experiment I provide
(theoretical) memory consumption of the quantized model.

## Results

The following table contains the results of most experiments. 
The results presented in the table accurately reflect the 
conducted experiments, as each of them was repeated several times (3-4).
I provide results with only one experiment with custom engine as it's only
getting worse.

| Model                	| Loss 	     | Accuracy<br>(%) 	     | CPU time<br>(s) 	 | Memory consumption<br>(Mb) 	|
|----------------------	|------------|-----------------------|-------------------|----------------------------	|
| Baseline             	| .389 	     | 90.89           	     | 48.9            	 | 1.117                      	|
| 8-bit torch PTQ      	| .391 	     | 90.67           	     | 33.6            	 | 0.279                      	|
| 16-bit torch (fp16)  	| **.389** 	 | **90.90**           	 | ---             	 | 0.559                      	|
| 8-bit custom PTQ     	| .391 	     | 90.57           	     | 34.2            	 | 0.279                      	|
| 4-bit custom PTQ     	| 1.86 	     | 34.08           	     | 32.7            	 | 0.139                      	|
| 2-bit custom PTQ     	| 2.30 	     | 10.03           	     | 32.9            	 | 0.07                       	|
| 16-bit custom engine 	| 4.25 	     | 09.89           	     | 361.8           	 | 0.557                      	|

As we can see, out custom 8-bit PTQ implementation based on PyTorch quantization API works 
a little worse than implementation from PyTorch tutorial. Our solution a bit
slower and achieves slightly inferior quality. Both solutions take almost 4 
times less memory than baseline and attain basically the same accuracy.

Surprisingly, 16-bit quantized model (with fp16 precision) achieves better quality
than baseline and takes 2 times less memory. However, this solution still can not be 
used on machine without GPU. The 4-bit quantized model shows substantial decline in
both loss and accuracy, although it is slightly faster compared to
all other models. Finally, the 2-bit quantized model demonstrates
random quality.

As mentioned earlier, experiments with custom quantization engine are
useless in terms of actual quantization: the loss is awful, the 
quality is random, and, finally, it works freaking slow.

# Relevant ideas
## Granulated quantization
Inspired by the paper [Quantizing deep convolutional networks for
efficient inference](https://arxiv.org/pdf/1806.08342.pdf).

Now we speak only about model weights. In above experiments we utilized `per tensor` quantization scheme.
That means, that for each tensor we selected one scale factor and one offset. However, for instance, each
convolutional kernel could learn its own weight distribution, which makes the scheme of finding a single
pair of quantization parameters for the entire tensor suboptimal. To address this issue I have 
[implemented](src/ptq/activation_observer.py) `PerChannelObserver` which utilizes `per channel` quantization
scheme. More precisely it collects statistics for each channel and after calibration computes a pair of 
quantization parameters for each channel. 

I applied that new Observer for model quantization and obtained the following results:

| Model                        | Loss | Accuracy<br>(%) | CPU time<br>(s) | Memory consumption<br>(Mb) |
|:-----------------------------|------|-----------------|-----------------|----------------------------|
| 8-bit per-channel (PyTorch)  | .384 | .9061           | 34.3            | 0.279                      |
| 8-bit per-channel (custom)   | .408 | .9034           | 32.4            | 0.279                      |
| 4-bit per-channel (custom)   | 1.81 | .372            | 33.2            | 0.139                      |
| 2-bit per-channel (custom)   | 2.30 | .1003           | 33.2            | 0.07                       |

Unexpectedly, we may observe increase in quality for 4-bit quantized model and slight decrease
for 8-bit model (both, model from PyTorch tutorial and the one quantized using `PerChannelObserver`).
Obviously, granulated quantization is relevant for this task.

## QAT
Previous experiments showed that even quantized to 8-bit model demonstrates
nearly the same accuracy as a baseline model. I guess, one reason of why is
it so being that I used automatic mixed precision during training, which made
model more resilient to variations in weight ranges. One another way to 
"prepare" model to quantization is QAT.

To perform QAT I used example from PyTorch tutorial (almost without changes).
I had to rewrite train function as I couldn't use QAT and AMP simultaneously.

As a result I obtained quantized to 8-bits model, which achieves **90.94%** accuracy.
That is the best quality of all observed models.
