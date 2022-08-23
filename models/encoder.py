import math
from typing import List
import paddle
from paddle import nn
import paddle.fft as fft
import numpy as np
from .dilated_conv import DilatedConvEncoder


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = paddle.full((B, T), True, dtype=paddle.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5, device=paddle.set_device("cpu")):
    return paddle.to_tensor(np.random.binomial(1, p, size=(B, T)), dtype=paddle.bool, place=device)


def calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


class BandedFourierLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201,
                 device=paddle.set_device("cpu"), name='0'):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands
        self.device = device

        self.num_freqs = self.total_freqs // self.num_bands + \
                         (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        # case: from other frequencies
        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight_{}".format(name), initializer=paddle.nn.initializer.KaimingUniform())
        self.weight = self.create_parameter(
            shape=[self.num_freqs, in_channels, out_channels],
            attr=weight_attr,
            dtype=paddle.float32,
            is_bias=False)
        fan_in, _ = calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bias_attr = paddle.framework.ParamAttr(
            name="linear_bias_{}".format(name),
            initializer=paddle.nn.initializer.Uniform(low=-bound, high=bound))
        self.bias = self.create_parameter(
            shape=[self.num_freqs, out_channels],
            attr=bias_attr,
            dtype=paddle.float32,
            is_bias=True)

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, axis=1)
        # output_fft = paddle.zeros((b, t // 2 + 1, self.out_channels), dtype=paddle.complex64)
        tmp = self._forward(input_fft)
        # print(0, self.start, self.end, output_fft.shape, tmp.shape)
        # output_fft[:, self.start:self.end] = tmp
        return fft.irfft(tmp, n=input.shape[1], axis=1)

    def _forward(self, input):
        output = paddle.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias


class CoSTEncoder(nn.Layer):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial',
                 device=paddle.set_device("cpu"),
                 name='0'):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.device = device

        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        self.tfd = nn.LayerList(
            [nn.Conv1D(output_dims, component_dims, k, padding=k-1) for k in kernels]
        )

        self.sfd = nn.LayerList(
            [BandedFourierLayer(output_dims, component_dims, b, 1,
                                length=length, name=name, device=self.device) for b in range(1)]
        )

    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        # x[~nan_mask] = 0

        x = self.input_fc(x)  # B x T x Ch
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.shape[0], x.shape[1], self.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.shape[0], x.shape[1])
        elif mask == 'all_true':
            mask = paddle.full((x.shape[0], x.shape[1]), True, dtype=paddle.bool)
        elif mask == 'all_false':
            mask = paddle.full((x.shape[0], x.shape[1]), False, dtype=paddle.bool)
        elif mask == 'mask_last':
            mask = paddle.full((x.shape[0], x.shape[1]), True, dtype=paddle.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose((0, 2, 1))  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T
        if tcn_output:
            return x.transpose((0, 2, 1))

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose((0, 2, 1)))  # b t d
        # trend = reduce(
        #     rearrange(trend, 'list b t d -> list b t d'),
        #     'list b t d -> b t d', 'mean'
        # )
        trend = paddle.mean(paddle.stack(trend), axis=0)
        x = x.transpose((0, 2, 1))  # B x T x Co

        season = []
        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]
        return trend, self.repr_dropout(season)
