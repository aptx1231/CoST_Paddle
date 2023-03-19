import math
from typing import List
import paddle
from paddle import nn
import paddle.fft as fft
import numpy as np
from paddle.nn.initializer import KaimingUniform
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
    def __init__(self, in_channels, out_channels, band=1, num_bands=1, length=201,
                 device=paddle.set_device("cpu"), name='0'):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        weight_shape = [self.total_freqs, in_channels, out_channels, 2]
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform())
        self._weight = self.create_parameter(weight_shape, weight_attr)

        bias_shape = [self.total_freqs, out_channels, 2]
        fan_in, _ = calculate_fan_in_and_fan_out(self._weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(low=-bound, high=bound))
        self._bias = self.create_parameter(bias_shape, bias_attr, is_bias=True)

    def forward(self, input):
        # input - (B, t, C)
        input = input.transpose((0, 2, 1))  # (B, C, t)
        batch_size, _, seq_len = input.shape
        out = paddle.fft.rfft(input, axis=-1)           # [B, C, t // 2 + 1]
        out = paddle.transpose(out, perm=[2, 0, 1])     # [t // 2 + 1, B, C]
        out = paddle.matmul(out, paddle.as_complex(self._weight))  # [t // 2 + 1, C, out] -> [t // 2 + 1, B, out]
        out = paddle.transpose(out, perm=[1, 0, 2])  # [B, t // 2 + 1, out]
        out = out + paddle.as_complex(self._bias)  # [B, t // 2 + 1, out]
        out = paddle.transpose(out, perm=[0, 2, 1])  # [B, out, t // 2 + 1]
        out = paddle.fft.irfft(out, seq_len, axis=-1)  # [B, out, t]
        out = out.transpose((0, 2, 1))  # (B, t, out)
        return out


def paddle_mask_fill(
    tensor: paddle.Tensor,
    mask: paddle.Tensor,
    value: float
) -> paddle.Tensor:
    """Fills elements of tensor with value where mask is True.
    Args:
        tensor(paddle.Tensor): The tensor to be masked.
        mask(paddle.Tensor): The boolean mask.
        value(float): The value to fill in with.
    Returns:
        paddle.Tensor: Output of function.
    """
    mask = paddle.expand_as(mask[:, :, None], tensor)
    cache = paddle.full(tensor.shape, value, tensor.dtype)
    return paddle.where(mask, cache, tensor)


class CoSTEncoder(nn.Layer):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial',
                 device=paddle.set_device("cpu")):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.device = device

        k = np.sqrt(1. / input_dims)
        weight_attr = bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(-k, k)
        )
        self.input_fc = nn.Linear(
            input_dims, hidden_dims,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )

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
                                length=length, device=self.device) for b in range(1)]
        )

    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims
        # nan_mask = ~x.isnan().any(axis=-1)
        # x[~nan_mask] = 0
        nan_mask = paddle.any(paddle.isnan(x), -1)
        x = paddle_mask_fill(x, nan_mask, 0)

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
        x = paddle_mask_fill(x, ~mask, 0)
        # x[~mask] = 0

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
