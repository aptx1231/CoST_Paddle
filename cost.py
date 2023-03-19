import math, random
from typing import Union, Callable, Optional, List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fft as fft
from paddle.io import Dataset, DataLoader, TensorDataset

import numpy as np

from models.encoder import CoSTEncoder
from utils import split_with_nan, centerize_vary_length_series, paddle_pad_nan


class PretrainDataset(Dataset):

    def __init__(self,
                 data,
                 sigma,
                 p=0.5,
                 multiplier=10):
        super().__init__()
        self.data = data
        self.p = p
        self.sigma = sigma
        self.multiplier = multiplier
        self.N, self.T, self.D = data.shape  # num_ts, time, dim

    def __getitem__(self, item):
        ts = self.data[item % self.N]
        return self.transform(ts), self.transform(ts)

    def __len__(self):
        return self.data.shape[0] * self.multiplier

    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + (paddle.randn(x.shape) * self.sigma)

    def scale(self, x):
        if random.random() > self.p:
            return x
        return x * (paddle.randn([x.shape[-1]]) * self.sigma + 1)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (paddle.randn([x.shape[-1]]) * self.sigma)


def freeze_batchnorm_statictis(layer):
    def freeze_bn(layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True

    layer.apply(freeze_bn)


class CoSTModel(nn.Layer):
    def __init__(self,
                 encoder_q: nn.Layer, encoder_k: nn.Layer,
                 kernels: List[int],
                 device: Optional[str] = 'cuda',
                 dim: Optional[int] = 128,
                 alpha: Optional[float] = 0.05,
                 K: Optional[int] = 65536,
                 m: Optional[float] = 0.999,
                 T: Optional[float] = 0.07):
        super().__init__()

        for name, param in encoder_q.named_parameters():
            print(str(name) + '\t' + str(param.shape) + '\t' + str(param.place))
        print('---------------------------------------------------------------')
        for name, param in encoder_k.named_parameters():
            print(str(name) + '\t' + str(param.shape) + '\t' + str(param.place))

        self.K = K
        self.m = m
        self.T = T
        self.device = device

        self.kernels = kernels

        self.alpha = alpha

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        # create the encoders
        self.head_q = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.head_k = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # param_k.data.copy_(param_q.data)  # initialize
            param_k.set_value(param_q)  # initialize
            param_k.stop_gradient = True  # not update by gradient
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            # param_k.data.copy_(param_q.data)  # initialize
            param_k.set_value(param_q)
            param_k.stop_gradient = True  # not update by gradient

        self.register_buffer('queue', F.normalize(paddle.randn((dim, K)), axis=0))
        self.register_buffer('queue_ptr', paddle.zeros([1], dtype=paddle.int64))

    def compute_loss(self, q, k, k_negs):
        # compute logits
        # positive logits: Nx1
        # l_pos = paddle.einsum('nc,cn->nn', [q, k.transpose((1, 0))]).unsqueeze(-1)
        # l_pos = paddle.matmul(q, k.transpose((1, 0))).diagonal().unsqueeze(-1)
        l_pos = paddle.sum(q * k, axis=1).unsqueeze(-1)
        # negative logits: NxK
        # l_neg = paddle.einsum('nc,ck->nk', [q, k_negs])
        l_neg = paddle.matmul(q, k_negs)

        # logits: Nx(1+K)
        logits = paddle.concat([l_pos, l_neg], axis=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators - first dim of each batch
        labels = paddle.zeros([logits.shape[0]], dtype=paddle.int64)
        loss = F.cross_entropy(logits, labels)

        return loss

    def convert_coeff(self, x, eps=1e-6):
        amp = paddle.sqrt((x.real() + eps).pow(2) + (x.imag() + eps).pow(2))
        phase = paddle.atan2(x.imag(), x.real() + eps)
        return amp, phase

    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.shape[0], z1.shape[1]
        z = paddle.concat([z1, z2], axis=0)  # 2B x T x C
        z = z.transpose((1, 0, 2))  # T x 2B x C
        sim = paddle.matmul(z, z.transpose((0, 2, 1)))  # T x 2B x 2B

        logits = paddle.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += paddle.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, axis=-1)

        # loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        loss = (logits[:, 0:B, B-1:2*B-1].mean() + logits[:, B:2*B, 0:B].mean()) / 2
        return loss

    def forward(self, x_q, x_k):
        # compute query features
        rand_idx = np.random.randint(0, x_q.shape[1])
        q_t, q_s = self.encoder_q(x_q)
        if q_t is not None:
            q_t = F.normalize(self.head_q(q_t[:, rand_idx]), axis=-1)

        # compute key features
        with paddle.no_grad():  # no gradient for keys
            self._momentum_update_key_encoder()  # update key encoder
            k_t, k_s = self.encoder_k(x_k)
            if k_t is not None:
                k_t = F.normalize(self.head_k(k_t[:, rand_idx]), axis=-1)

        loss = 0

        loss += self.compute_loss(q_t, k_t, self.queue.clone().detach())
        self._dequeue_and_enqueue(k_t)

        q_s = F.normalize(q_s, axis=-1)
        _, k_s = self.encoder_q(x_k)
        k_s = F.normalize(k_s, axis=-1)

        q_s_freq = fft.rfft(q_s, axis=1)
        k_s_freq = fft.rfft(k_s, axis=1)
        q_s_amp, q_s_phase = self.convert_coeff(q_s_freq)
        k_s_amp, k_s_phase = self.convert_coeff(k_s_freq)

        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp) + \
                        self.instance_contrastive_loss(q_s_phase, k_s_phase)
        loss += (self.alpha * (seasonal_loss/2))

        return loss

    # @paddle.no_grad()
    # def _momentum_update_key_encoder(self):
    #     """
    #     Momentum update for key encoder
    #     """
    #     for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
    #         param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
    #     for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
    #         param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @paddle.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update for key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            param_k.stop_gradient = True
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            param_k.stop_gradient = True

    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose([1, 0])

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr


class CoST:
    def __init__(self,
                 input_dims: int,
                 kernels: List[int],
                 alpha: bool,
                 max_train_length: int,
                 output_dims: int = 320,
                 hidden_dims: int = 64,
                 depth: int = 10,
                 device: 'str' ='cuda',
                 lr: float = 0.001,
                 batch_size: int = 16,
                 after_iter_callback: Union[Callable, None] = None,
                 after_epoch_callback: Union[Callable, None] = None):

        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length

        if kernels is None:
            kernels = []

        self.net_q = CoSTEncoder(
            input_dims=input_dims, output_dims=output_dims,
            kernels=kernels,
            length=max_train_length,
            hidden_dims=hidden_dims, depth=depth,
        ).to(self.device)

        self.net_k = CoSTEncoder(
            input_dims=input_dims, output_dims=output_dims,
            kernels=kernels,
            length=max_train_length,
            hidden_dims=hidden_dims, depth=depth,
        ).to(self.device)

        self.cost = CoSTModel(
            self.net_q,
            self.net_k,
            kernels=kernels,
            dim=self.net_q.component_dims,
            alpha=alpha,
            K=256,
            device=self.device,
        ).to(self.device)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        train_data = np.nan_to_num(train_data)

        multiplier = 1 if train_data.shape[0] >= self.batch_size \
            else math.ceil(self.batch_size / train_data.shape[0])
        train_dataset = PretrainDataset(paddle.to_tensor(
            train_data, dtype=paddle.float32, place=self.device), sigma=0.5, multiplier=multiplier)
        train_loader = DataLoader(train_dataset, batch_size=min(
            self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)

        optimizer = paddle.optimizer.Momentum(
            parameters=[p for p in self.cost.parameters() if p.stop_gradient is False],
            learning_rate=self.lr, momentum=0.9, weight_decay=1e-4)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x_q, x_k = map(lambda x: x, batch)
                # x_q, x_k = map(lambda x: x.to(self.device), batch)
                if self.max_train_length is not None and x_q.shape[1] > self.max_train_length:
                    window_offset = np.random.randint(x_q.shape[1] - self.max_train_length + 1)
                    x_q = x_q[:, window_offset : window_offset + self.max_train_length]
                    x_k = x_k[:, window_offset : window_offset + self.max_train_length]

                loss = self.cost(x_q, x_k)

                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

                if n_iters is not None:
                    adjust_learning_rate(optimizer, self.lr, self.n_iters, n_iters)
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

            if n_epochs is not None:
                adjust_learning_rate(optimizer, self.lr, self.n_epochs, n_epochs)
            
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out_t, out_s = self.net_q(x)  # l b t d  .to(self.device, non_blocking=True)
        out = paddle.concat([out_t[:, -1], out_s[:, -1]], axis=-1)
        return out.unsqueeze(1).cpu()  # rearrange(out.cpu(), 'b d -> b () d')
    
    def encode(self, data, mode, mask=None, encoding_window=None,
               casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        if mode == 'forecasting':
            encoding_window = None
            slicing = None
        else:
            raise NotImplementedError(f"mode {mode} has not been implemented")

        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net_q.training
        self.net_q.eval()

        tensor_data = paddle.to_tensor(data, dtype=paddle.float32, place=self.device)
        dataset = TensorDataset([tensor_data])
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with paddle.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = paddle_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    paddle.concat(calc_buffer, axis=0),
                                    mask,
                                    slicing=slicing,
                                    encoding_window=encoding_window
                                )
                                reprs += paddle.split(out, out.shape[0])
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                paddle.concat(calc_buffer, axis=0),
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs += paddle.split(out, out.shape[0])
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = paddle.concat(reprs, axis=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose((0, 2, 1)).contiguous(),
                            kernel_size = out.shape[1],
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = paddle.concat(output, axis=0)

        if org_training:
            self.net_q.train()
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        paddle.save(self.net_q.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = paddle.load(fn)
        self.net_q.set_state_dict(state_dict)


def adjust_learning_rate(optimizer, lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    optimizer.set_lr(lr)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
