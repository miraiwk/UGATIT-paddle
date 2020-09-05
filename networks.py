#coding:utf-8
from paddle import fluid
from paddle.fluid.dygraph import *
from paddle.fluid.initializer import Constant
import paddle.fluid.layers as L
import numpy as np
import ops.paddle_nn as nn
from ops.paddle_nn import var

class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        super(ResnetGenerator, self).__init__()
        assert(n_blocks >= 0)
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = [nn.ReflectionPad2d(3),
                nn.MyConv2D(input_nc, ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(),
                ]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.MyConv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU()]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = nn.MyLinear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = nn.MyLinear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = nn.MyConv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True)
        self.relu = nn.ReLU()

        # Gamma, Beta block
        if self.light:
            FC = [nn.MyLinear(ngf * mult, ngf * mult, bias_attr=False),
                  nn.ReLU(),
                  nn.MyLinear(ngf * mult, ngf * mult, bias_attr=False),
                  nn.ReLU()]
        else:
            FC = [nn.MyLinear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False),
                  nn.ReLU(),
                  nn.MyLinear(ngf * mult, ngf * mult, bias_attr=False),
                  nn.ReLU()]
        self.gamma = nn.MyLinear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = nn.MyLinear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.MyConv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU()]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.MyConv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False),
                     nn.Tanh()]

        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        gap = L.adaptive_pool2d(x, 1, pool_type='avg')
        gap_logit = self.gap_fc(L.reshape(gap, (x.shape[0], -1)))
        gap_weight = self.gap_fc.weight
        gap = x * L.unsqueeze(gap_weight, (2, 3))

        gmp = L.adaptive_pool2d(x, 1, pool_type='max')
        gmp_logit = self.gmp_fc(L.reshape(gmp, (x.shape[0], -1)))
        gmp_weight = self.gmp_fc.weight
        gmp = x * L.unsqueeze(gmp_weight, (2, 3))

        cam_logit = L.concat([gap_logit, gmp_logit], 1)
        x = L.concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = L.reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = L.adaptive_pool2d(x, 1, pool_type='avg')
            x_ = self.FC(L.reshape(x_, (x_.shape[0], -1)))
        else:
            x_ = self.FC(L.reshape(x, (x.shape[0], -1)))
        gamma, beta = self.gamma(x_), self.beta(x_)


        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap

class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.MyConv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(),
                       ]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.MyConv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       nn.InstanceNorm2d(dim)
                       ]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter(
                attr=True,
                shape=(1, num_features, 1, 1),
                dtype='float32',
                default_initializer=Constant(0.9),
                is_bias=False
                )
        self.num_features = num_features

    def forward(self, input, gamma, beta):
        rho_ = L.clip(self.rho, min=0, max=1)
        in_mean = L.reduce_mean(input, dim=[2, 3], keep_dim=True)
        in_var = var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / L.sqrt(in_var + self.eps)
        ln_mean = L.reduce_mean(input, dim=[1, 2, 3], keep_dim=True) 
        ln_var = var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / L.sqrt(ln_var + self.eps)
        out = rho_ * out_in + (1 - rho_) * out_ln
        out = out * L.unsqueeze(gamma, axes=[2, 3]) + L.unsqueeze(beta, axes=[2, 3])

        return out


class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter(
                attr=True,
                shape=(1, num_features, 1, 1),
                dtype='float32',
                default_initializer=Constant(0.0),
                is_bias=False
                )
        self.gamma = self.create_parameter(
                attr=True,
                shape=(1, num_features, 1, 1),
                dtype='float32',
                default_initializer=Constant(1.0),
                is_bias=False
                )
        self.beta = self.create_parameter(
                attr=True,
                shape=(1, num_features, 1, 1),
                dtype='float32',
                default_initializer=Constant(0.0),
                is_bias=False
                )
        self.num_features = num_features

    def forward(self, input):
        rho_ = L.clip(self.rho, min=0, max=1)
        in_mean = L.reduce_mean(input, dim=[2, 3], keep_dim=True)
        in_var = var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / L.sqrt(in_var + self.eps)
        ln_mean = L.reduce_mean(input, dim=[1, 2, 3], keep_dim=True) 
        ln_var = var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / L.sqrt(ln_var + self.eps)
        out = rho_ * out_in + (1 - rho_) * out_ln
        out = out * self.gamma + self.beta

        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.MyConv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU()

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.MyConv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.SpectralNormConv(input_nc, ndf, filter_size=4, stride=2, padding=0, bias_attr=True),
                 nn.LeakyReLU(0.2)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.SpectralNormConv(ndf * mult, ndf * mult * 2,
                      filter_size=4, stride=2, padding=0, bias_attr=True),
                      nn.LeakyReLU(0.2)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.SpectralNormConv(ndf * mult, ndf * mult * 2,
                  filter_size=4, stride=1, padding=0, bias_attr=True),
                  nn.LeakyReLU(0.2)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.SpectralNormLinear(ndf * mult, 1, bias_attr=False)
        self.gmp_fc = nn.SpectralNormLinear(ndf * mult, 1, bias_attr=False)
        self.conv1x1 = nn.MyConv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.SpectralNormConv(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False)

        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = L.adaptive_pool2d(x, 1, pool_type='avg')
        gap_logit = self.gap_fc(L.reshape(gap, (x.shape[0], -1)))
        gap_weight = self.gap_fc.weight_orig
        gap = x * L.unsqueeze(gap_weight, (2, 3))

        gmp = L.adaptive_pool2d(x, 1, pool_type='max')
        gmp_logit = self.gmp_fc(L.reshape(gmp, (x.shape[0], -1)))
        gmp_weight = self.gmp_fc.weight_orig
        gmp = x * L.unsqueeze(gmp_weight, (2, 3))

        cam_logit = L.concat([gap_logit, gmp_logit], 1)

        x = L.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = L.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap
