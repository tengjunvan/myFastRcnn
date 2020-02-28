# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:19:09 2020

@author: tengjunwan
"""

from collections import namedtuple
from string import Template

import cupy, torch
import cupy as cp
import torch as t
from torch.autograd import Function

from model.utils.roi_cupy import kernel_backward, kernel_forward

Stream = namedtuple('Stream', ['ptr'])


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K


class RoI(Function):
    """这个RoI是roi pooling layer的真实类，
    """
    def __init__(self, outh, outw, spatial_scale):
        self.forward_fn = load_kernel('roi_forward', kernel_forward)
        self.backward_fn = load_kernel('roi_backward', kernel_backward)
        self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale

    def forward(self, x, rois):
        # NOTE: MAKE SURE input is contiguous too
        x = x.contiguous()  # 这里的输入是(b, C, W, H)
        rois = rois.contiguous()  # 这里是(N,5)
        self.in_size = B, C, H, W = x.size()
        self.N = N = rois.size(0)
        output = t.zeros(N, C, self.outh, self.outw).cuda()
        self.argmax_data = t.zeros(N, C, self.outh, self.outw).int().cuda()
        self.rois = rois
        args = [x.data_ptr(), rois.data_ptr(),
                output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.spatial_scale, C, H, W,
                self.outh, self.outw,
                output.numel()]
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        self.forward_fn(args=args,
                        block=(CUDA_NUM_THREADS, 1, 1),
                        grid=(GET_BLOCKS(output.numel()), 1, 1),
                        stream=stream)
        return output

    def backward(self, grad_output):
        ##NOTE: IMPORTANT CONTIGUOUS
        # TODO: input
        grad_output = grad_output.contiguous()
        B, C, H, W = self.in_size
        grad_input = t.zeros(self.in_size).cuda()
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        args = [grad_output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.rois.data_ptr(),
                grad_input.data_ptr(),
                self.N, self.spatial_scale, C, H, W, self.outh, self.outw,
                grad_input.numel()]
        self.backward_fn(args=args,
                         block=(CUDA_NUM_THREADS, 1, 1),
                         grid=(GET_BLOCKS(grad_input.numel()), 1, 1),
                         stream=stream
                         )
        return grad_input, None


class RoIPooling2D(t.nn.Module):
    """这里是虚假的roi pooling layer类，封装了下上面那个真实的类
    但这个类更加适合nn.Module的使用习惯
    """
    def __init__(self, outh, outw, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.RoI = RoI(outh, outw, spatial_scale)

    def forward(self, x, rois):
        return self.RoI(x, rois)


def test_roi_module():
    ## fake data###
    B, N, C, H, W, PH, PW = 2, 8, 4, 32, 32, 7, 7

    bottom_data = t.randn(B, C, H, W).cuda()  # (2, 4, 32, 32)
    bottom_rois = t.randn(N, 5)  # (8, 5)
    bottom_rois[:int(N / 2), 0] = 0  # 第一维4表示8张图的4个编号是0，4个是1
    bottom_rois[int(N / 2):, 0] = 1
    bottom_rois[:, 1:] = (t.rand(N, 4) * 100).float()  #后面四个维度是坐标随机
    # 这样完成了输入数据构造，一个是网络前端(feature层)的输入data
    # 另一个是roi的输入rois（N,5），这里需要说明的是，我的代码没有考虑到多batch的时候，
    # 即data的batch维度是1，rois第一维度永远是0
    bottom_rois = bottom_rois.cuda()
    spatial_scale = 1. / 16
    outh, outw = PH, PW

    # pytorch version
    module = RoIPooling2D(outh, outw, spatial_scale)  # 初始化7，7，1/16 
    x = bottom_data.requires_grad_()
    rois = bottom_rois.detach()

    output = module(x, rois)  # 输出是(8, 4, 7, 7),其中8是2*4
    print(output.shape)
    output.sum().backward()

    def t2c(variable):
        npa = variable.data.cpu().numpy()
        return cp.array(npa)

    def test_eq(variable, array, info):
        cc = cp.asnumpy(array)
        neq = (cc != variable.data.cpu().numpy())
        assert neq.sum() == 0, 'test failed: %s' % info

    # chainer version,if you're going to run this
    # pip install chainer 
    import chainer.functions as F
    from chainer import Variable
    x_cn = Variable(t2c(x))

    o_cn = F.roi_pooling_2d(x_cn, t2c(rois), outh, outw, spatial_scale)
    test_eq(output, o_cn.array, 'forward')
    F.sum(o_cn).backward()
    test_eq(x.grad, x_cn.grad, 'backward')
    print('test pass')


def dimension_test():
    """这个是我写的测试输入维度和输出维度的
    """
    B, N, C, H, W, PH, PW = 1, 1, 1, 4, 4, 1, 1

    bottom_data = t.randn(B, C, H, W).cuda()  # (1, 1, 4, 4)
    bottom_rois = t.randn(N, 5)  # (1, 5)
    bottom_rois[:, 0] = 0  
    bottom_rois[:, 1:] = t.Tensor([15, 15, 16, 16]).float()  #后面四个维度是坐标随机
    # 这样完成了输入数据构造，一个是网络前端(feature层)的输入data
    # 另一个是roi的输入rois（N,5），这里需要说明的是，我的代码没有考虑到多batch的时候，
    # 即data的batch维度是1，rois第一维度永远是0
    bottom_rois = bottom_rois.cuda()
    spatial_scale = 1. / 16
    outh, outw = PH, PW

    # pytorch version
    module = RoIPooling2D(outh, outw, spatial_scale)  # 初始化7，7，1/16 
    x = bottom_data.requires_grad_()
    rois = bottom_rois.detach()

    output = module(x, rois)  # 输出是(8, 4, 7, 7),其中8是2*4
    print(output.shape)
    print(output)
    print(x)
    output.sum().backward()
    print(x.grad)
    
    
    
    
    
    
if __name__ == "__main__":
    test_roi_module()
    dimension_test()