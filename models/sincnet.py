import math
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple


class SincConv(nn.Module):
    """This function implements SincConv (SincNet).
    M. Ravanelli, Y. Bengio, "Speaker Recognition from raw waveform with
    SincNet", in Proc. of  SLT 2018 (https://arxiv.org/abs/1808.00158)
    Arguments
    ---------
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    out_channels : int
        It is the number of output channels.
    kernel_size: int
        Kernel size of the convolutional filters.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    groups : int
        This option specifies the convolutional groups. See torch.nn
        documentation for more information.
    bias : bool
        If True, the additive bias b is adopted.
    sample_rate : int,
        Sampling rate of the input signals. It is only used for sinc_conv.
    min_low_hz : float
        Lowest possible frequency (in Hz) for a filter. It is only used for
        sinc_conv.
    min_low_hz : float
        Lowest possible value (in Hz) for a filter bandwidth.
    Example
    -------
    >>> inp_tensor = torch.rand([10, 16000])
    >>> conv = SincConv(input_shape=inp_tensor.shape, out_channels=25, kernel_size=11)
    >>> out_tensor = conv(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 16000, 25])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding="same",
        padding_mode="reflect",
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # input shape inference
        if input_shape is None and self.in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if self.in_channels is None:
            self.in_channels = self._check_input_shape(input_shape)

        if self.out_channels % self.in_channels != 0:
            raise ValueError(
                "Number of output channels must be divisible by in_channels"
            )

        # Initialize Sinc filters
        self._init_sinc_conv()

    def forward(self, x):
        """Returns the output of the convolution.
        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.
        """
        # x = x.permute(0,2,1)
        # print(x.shape)
        # # x = x.transpose(1, -1)
        # print(x.shape)
        self.device = x.device

        unsqueeze = x.ndim == 2
        if unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got %s."
                % (self.padding)
            )

        print(self.low_hz_, self.low_hz_.shape)
        print(self.band_hz_, self.band_hz_.shape)

        sinc_filters = self._get_sinc_filters()


        # print(sinc_filters.shape)

        # off = F.sigmoid(self.offset)
        # # print(off, off.shape)
        # off = (off*self.kernel_size) - self.kernel_size/2
        # print(off, off.shape)

        # print(x.shape)
        # print(sinc_filters.shape)

        # x = x.permute(0,2,1)
        # print(x.shape)
        # test = F.pad(x, (10, 0), "constant", 0)
        # print(test.shape)

        # for samp, offsetter in zip(x, off):
        #     print(samp, offsetter)
        #     print(samp.shape, offsetter.shape)

        print("INPUT SHAPE", x.shape)

        # for idx in range(self.out_channels):
        #     print(x[:,0,:].shape)
            # print(x[idx], off[idx])
            # print(x[idx].shape, off[idx].shape)
            # print(F.pad(x[idx], (10, 0), "constant", 0).shape)


        wx = F.conv1d(
            x,
            sinc_filters,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            # groups=self.in_channels,
        )

        # print('Wx', wx.shape)

        # if unsqueeze:
        #     wx = wx.squeeze(1)

        # wx = wx.transpose(1, -1)

        return wx

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels.
        """

        if len(shape) == 2:
            in_channels = 1
        elif len(shape) == 3:
            in_channels = shape[-1]
        else:
            raise ValueError(
                "sincconv expects 2d or 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )
        return in_channels

    def _get_sinc_filters(self,):
        """This functions creates the sinc-filters to used for sinc-conv.
        """
        # Computing the low frequencies of the filters
        low = self.min_low_hz + torch.abs(self.low_hz_)

        # Setting minimum band and minimum freq
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        # Passing from n_ to the corresponding f_times_t domain
        self.n_ = self.n_.to(self.device)
        self.window_ = self.window_.to(self.device)
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Left part of the filters.
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low))
            / (self.n_ / 2)
        ) * self.window_

        # Central element of the filter
        band_pass_center = 2 * band.view(-1, 1)

        # Right part of the filter (sinc filters are symmetric)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        # Combining left, central, and right part of the filter
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        # Amplitude normalization
        band_pass = band_pass / (2 * band[:, None])

        # Setting up the filter coefficients
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        # print("filter shape", filters.shape)


        off = F.sigmoid(self.offset).to(self.device)
        off = (off*self.kernel_size)
        print(off, self.offset, self.offset.grad)

        # print(torch.floor(off))
        # print(F.sigmoid(100*torch.remainder(off, 1)))

        rounding = torch.floor(off) + F.sigmoid(100*torch.remainder(off, 1))

        # print(rounding)

        bigfilter = torch.zeros(self.out_channels, 2, self.kernel_size*2).to(self.device)
        # print("big filter shape", bigfilter.shape)

        for idx in range(bigfilter.shape[0]):
            # print(bigfilter[idx][0].shape)
            print(rounding[idx][0].long())
            print(bigfilter[idx][0][rounding[idx][0].long()])
            # print(filters[idx][0].shape)
            bigfilter[idx][0][int(rounding[idx][0]):int(rounding[idx][0])+self.kernel_size] = filters[idx][0]
            bigfilter[idx][1][int(rounding[idx][1]):int(rounding[idx][1])+self.kernel_size] = filters[idx][0]

        # print(bigfilter, "BIGFILTER")

        return bigfilter

    def _init_sinc_conv(self):
        """Initializes the parameters of the sinc_conv layer."""

        # Initialize filterbanks such that they are equally spaced in Mel scale
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = torch.linspace(
            self._to_mel(self.min_low_hz),
            self._to_mel(high_hz),
            self.out_channels + 1,
        )

        hz = self._to_hz(mel)

        # Filter lower frequency and bands
        self.low_hz_ = hz[:-1].unsqueeze(1)
        self.band_hz_ = (hz[1:] - hz[:-1]).unsqueeze(1)

        # Maiking freq and bands learnable
        self.low_hz_ = nn.Parameter(self.low_hz_)
        self.band_hz_ = nn.Parameter(self.band_hz_)


        self.offset = torch.zeros((self.out_channels, self.in_channels))

        self.offset = nn.Parameter(self.offset, requires_grad=True)

        # Hamming window
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )
        self.window_ = 0.54 - 0.46 * torch.cos(
            2 * math.pi * n_lin / self.kernel_size
        )

        # Time axis  (only half is needed due to symmetry)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )

    def _to_mel(self, hz):
        """Converts frequency in Hz to the mel scale.
        """
        return 2595 * np.log10(1 + hz / 700)

    def _to_hz(self, mel):
        """Converts frequency in the mel scale to Hz.
        """
        return 700 * (10 ** (mel / 2595) - 1)

    def _manage_padding(
        self, x, kernel_size: int, dilation: int, stride: int,
    ):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.
        """

        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x