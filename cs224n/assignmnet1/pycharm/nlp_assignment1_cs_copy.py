import numpy as np
def softmax(x):
    orig_shape=x.shape
    print('shape',orig_shape)
    print('shape_len',len(x.shape))

    if len(x.shape)>1:
        exp_minmax=lambda x:np.exp(x-np.max(x))  #分子部分
        denom=lambda x:1.0/np.sum(x)  #分母部分
        #参考notebook中sumply部分。E:\study_series\tensorflow\nlp\wcs\assignment1
        x=np.apply_along_axis(exp_minmax,1,x) #样本是按照行存储的，把每个x都加起来，应该是按照列走，及1，
        denominator=np.apply_along_axis(denom,1,x)
        if len(denominator.shape)==1:
            denominator=denominator.reshape((denominator.shape[0],1))
        x=x*denominator
    else:
        print('here')
        x_max=np.max(x)
        x=x-x_max
        numberator=np.exp(x)
        denominator=1.0/np.sum(numberator)
        x=numberator.dot(denominator)
    assert x.shape==orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print ('test1',test1)
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print ('test2',test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print ('test3',test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print( "You should be able to verify these results by hand!\n")

def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print ("Running your tests...")
    ### YOUR CODE HERE
    raise NotImplementedError  #这样子就会报错....
    ### END YOUR CODE

if __name__=="__main__":
    test_softmax_basic()
    test_softmax()