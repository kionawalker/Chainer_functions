import chainer
from chainer import backends
from chainer import function_node
# from chainer.utils import type_chec


class MySpace2Depth(chainer.functions.Space2Depth):
    def forward(self, inputs):
        X, = inputs
        xp = backends.cuda.get_array_module(X)
        bsize, c, a, b = X.shape
        X = xp.reshape(
            X, (bsize, c, a // self.r, self.r, b // self.r, self.r))
        # print(X)
        X = xp.transpose(X, (1, 3, 5, 0, 2, 4))
        # print(X)
        X = xp.reshape(
            X, (bsize, self.r ** 2 * c, a // self.r, b // self.r))
        return X,


def space2depth(X, r):
    return MySpace2Depth(r).apply((X,))[0]