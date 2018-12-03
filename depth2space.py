import chainer
from chainer import backends
from chainer import function_node
# from chainer.utils import type_chec
from myfunk import space2depth as my
import numpy


class MyDepth2Space(chainer.functions.Depth2Space):
    def forward(self, inputs):
        X, = inputs
        xp = backends.cuda.get_array_module(X)
        bsize, c, a, b = X.shape
        c //= self.r ** 2

        if xp is numpy:
            # # These codes run faster on CPU than below `else` block codes.
            # X = xp.transpose(X, (0, 2, 3, 1))
            # X = xp.reshape(X, (bsize, a, b, self.r, self.r, c))
            # X = xp.transpose(X, (0, 1, 3, 2, 4, 5))
            # X = xp.reshape(X, (bsize, a * self.r, b * self.r, c))
            # X = xp.transpose(X, (0, 3, 1, 2))
            # 元の実装とchannelの位置を変更
            X = xp.reshape(X, (bsize, c, self.r, self.r, a, b))
            X = xp.transpose(X, (0, 1, 4, 2, 5, 3))
            X = xp.reshape(X, (bsize, c, a * self.r, b * self.r))
        else:
            # 元の実装とchannelの位置を変更
            X = xp.reshape(X, (bsize, c, self.r, self.r, a, b))
            X = xp.transpose(X, (0, 1, 4, 2, 5, 3))
            X = xp.reshape(X, (bsize, c, a * self.r, b * self.r))
        return X,

    def backward(self, indexes, grad_outputs):
            gy, = grad_outputs
            gy = my.space2depth(gy, self.r)
            return gy,


def depth2space(X, r):
    return MyDepth2Space(r).apply((X,))[0]