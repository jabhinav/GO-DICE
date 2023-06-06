import scipy.signal


def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter(b=[1], a=[1, -gamma], x=x[::-1], axis=0)[::-1]