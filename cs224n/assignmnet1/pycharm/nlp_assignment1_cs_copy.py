import numpy as np
def softmax(x):
    orig_shape=x.shape
    print('shape',orig_shape)
    print('shape_len',len(x.shape))

    if len(x.shape)>1:
        exp_minmax=lambda x:np.exp(x-np.max(x))
    else:
        print('here')
        x_max=np.max(x)
        x=x-x_max
        numberator=np.exp(x)
        denominator=1.0/np.sum(numberator)
        x=numberator.dot(denominator)
    assert x.shape==orig_shape
    return x
print(softmax(np.array([1,2])))
