# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:32:44 2017

@author: Yao You

This script contains cost function calculations 
"""
import numpy as np
import scipy.io
import pylab as pl
from PIL import Image
import fnmatch
import os

def nnCostFunction(nn_params, layersizes, X, y, reg_para):
    """ Calculates the cost function based on the following inputs
    nn_params: neural network parameters, in long 1D array format
    layersizes: size of each layer, with layersizes[0] the number of inputs
                and layersizes[end] the number of outputs, 1s not included        
    X: input training examples, size (m, layersizes[0])
    y: answer for training examples, size (m)
    reg_para: parameter for regularization
    """
    
    """useful constants"""
    m = X.shape[0]
    L = layersizes.size
    y = y.reshape(m, 1)
    
    """convert y into a matrix"""
    if layersizes[-1] > 2:
        vec_y = np.zeros(shape=(m, layersizes[-1]))
        for i in range(0, m):
            vec_y[i, y[i] - 1] = 1
    else:
        vec_y = y
    """setting initial parameters"""
    nobias_params = np.zeros(1)
    inputs = {}
    acts = {}
    acts[0] = X
    
    for i in range(0, L-1):
        """ reshape the parameters into proper dimensions"""
        theta = np.array(nn_params[i])
        """ append the no biased nn paras into the collection"""
        nobias_params = np.append( nobias_params, np.append(
                        theta[:, 1:-1].ravel(), theta[:, -1]) )
        """calculates the layer values"""
        inputs[i] = np.concatenate((np.ones(shape = (m, 1)), acts[i]), axis=1)
        acts[i+1] = np.array(sigmoid(np.dot(inputs[i], theta.T)))
    """nobias portion of cost"""
    nbcost = (vec_y * np.log(acts[L-1])) + \
             ((1.0 - vec_y) * np.log(1.0 - acts[L-1]))
    """parameter cost"""
    bcost = np.dot(nobias_params.T, nobias_params)
    """ total cost """
    J = -nbcost.sum().sum() / m + reg_para * bcost / m / 2
    
    """ now move onto gradients """
    deltas = {}
    deltas[L-1] = (acts[L-1] - vec_y).T
    """ again fetch the parameters from the unrolled vector """
    theta = np.array(nn_params[L-2])
    theta[:, 0] = 0.0
    grad = {}
    grad[L-2] = np.dot(deltas[L-1], inputs[L-2]) / m + reg_para * theta / m
    for i in range(L-2, 0, -1):
        theta = np.array(nn_params[i])
        deltas[i] = np.dot(theta.T, deltas[i+1]) * inputs[i].T * (1 - inputs[i].T)
        theta = np.array(nn_params[i-1])
        theta[:, 0] = 0.0
        grad[i-1] = np.dot(deltas[i][1:layersizes[i] + 1, :], inputs[i-1]) / m + \
            reg_para * theta / m
        deltas[i] = deltas[i][1:layersizes[i] + 1, :]
    
    return {'cost': J, 'grad' : grad}
 

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidgrad(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def debugInitializeWeights(layers):
    """ use sine function to generate test parameters and input """
    W = {}
    L = layers.size
    for i in range(1, L):
        W[i-1] = np.sin(np.arange(1, 1 + layers[i] * (1 + layers[i-1]))).reshape(\
                         layers[i], 1 + layers[i-1]) / 10
    return W
 
def checkNNGradients(reg_para):
    """ Using numerical approximation for gradients to check the back
    propagation algorithm """
    layersizes = np.array([3, 5, 3])
    m = 5
    nn_params = debugInitializeWeights(layersizes)
    Xin = debugInitializeWeights(np.array([layersizes[0]-1, m]))
    Xin = np.array(Xin[0])
    yout = np.array(1 + np.mod(np.arange(1, m+1), layersizes[-1]))
    costngrad = nnCostFunction(nn_params, layersizes, Xin, yout, reg_para)
    grad = costngrad['grad']
    print (grad[0])
    """ calculate numerical gradients 
    numgrad = numericalGradients(nn_params)
    """
    
def testCF():
    """ use preset data to test the cost function and gradient calculations"""
    layersizes = [400, 25, 10]
    mat = scipy.io.loadmat('ex4data1.mat')
    nn_params = scipy.io.loadmat('ex4weights.mat')
    reg_para = 1.0
    
    Xin = np.array(mat['X'])
    yout = np.array(mat['y'])
    testparams = scipy.io.loadmat('ex4weights.mat')
    nn_params = {}
    nn_params[0] = np.array(testparams['Theta1'])
    nn_params[1] = np.array(testparams['Theta2'])
    """nn_params = np.append(theta1, theta2)"""
    
    costngrad = nnCostFunction(nn_params, np.array(layersizes), Xin, yout, reg_para)
    print(costngrad['cost'])
    checkNNGradients(1.0)


def randInitWeights(layersizes, epsilon):
    W = {}
    for i in range(0, layersizes.size - 1):
        W[i] = np.random.uniform(-epsilon, epsilon, \
                 (layersizes[i + 1] * (layersizes[i] + 1)) ) \
                 .reshape(layersizes[i + 1], layersizes[i] + 1)
    return W


def gradientDecend(X, y, layersizes, alpha, reg_para, epsilon, niter, small):
    """use gradient decent to solve for nn parameters """
    nn_params = randInitWeights(layersizes, epsilon)
    J = np.zeros(shape = (niter, 1))
    
    for i in range(0, niter):
        costngrad = nnCostFunction(nn_params, layersizes, X, y, reg_para)
        J[i] = costngrad['cost']
        grad = costngrad['grad']
        for k in range(0, len(nn_params)):
            nn_params[k] = nn_params[k] - alpha * grad[k]
        """stop at sufficiently small cost"""
        if (J[i] < small):
            break

    return {'costs' : J, 'parameters' : nn_params}


def nnPredict(X, nn_params):    
    y = X
    for nn_para in nn_params:
        y = sigmoid( np.dot(np.concatenate((np.ones(shape=(y.shape[0], 1)), y),\
                                           axis=1), \
                           nn_params[nn_para].T) )
    if (y.shape[1] == 1):
        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        return y
    else:
        return y.argmax(axis=1) + 1

def testnn():
    layersizes = np.array([400, 25, 10])
    mat = scipy.io.loadmat('ex4data1.mat')
    reg_para = 1.0
    alpha = 1
    epsilon = 0.12
    niter = 500
    small = 0.1
    
    X = np.array(mat['X'])
    y = np.array(mat['y'])
    
    return gradientDecend(X, y, layersizes, alpha, reg_para, epsilon, \
                              niter, small)

def readdata(rspath, imgsize):
    npixel = imgsize[0]*imgsize[1]
    ppath = os.path.join(rspath, 'person')
    nppath = os.path.join(rspath, 'no_person')
    
    imgp = readImgs(ppath, imgsize)
    imgnp = readImgs(nppath, imgsize)
    num_ptraining = imgp.shape[0]
    num_nptraining = imgnp.shape[0]
    imgs = np.concatenate((imgp, imgnp))
    
    """ predefined classes"""
    classes = np.zeros(shape = (num_ptraining + num_nptraining, 1))
    classes[0:num_ptraining] = 1
    classes[num_ptraining:num_ptraining + num_nptraining] = 0   
    
    """ setting layers up """
    layersizes = np.array([npixel, 100, 100, 1])
    
    return {'X': imgs / 255.0, 'y': classes, 'layersizes': layersizes}

def readImgs(p2img, imgsize):
    npixel = imgsize[0]*imgsize[1]
    
    imgnames = fnmatch.filter(os.listdir(p2img), '*.jpg')
    nimg = len(imgnames)
    imgs = np.zeros(shape = (nimg, npixel))
    """ loading images with person(s)"""
    for i in range(0, nimg):
        imgs[i,:] = np.array(Image.open(os.path.join(p2img, imgnames[i]))\
            .resize(imgsize).convert('L').getdata()).ravel()
    return imgs

def main():
    """Test the cost function and gradient calculations
    """   
    """response = testnn()"""
    
    """ load data from pre-classified folders """
    rspath = os.path.join('..', 'resource')
    imgsize = (128, 72)
    inputs = readdata(rspath, imgsize)
    X = inputs['X']
    y = inputs['y']
    layersizes = inputs['layersizes']
    reg_para = 1.0
    alpha = 1
    epsilon = 0.12
    niter = 1000
    small = 0.1
    response = gradientDecend(X, y, layersizes, \
                              alpha, reg_para, epsilon, niter, small)
    pl.plot(range(0, niter), response['costs'])
    print(response['costs'][-1])
    py = nnPredict(X, response['parameters']).reshape(y.shape)
    print((py == y).sum() * 1.0 / y.size * 1.0)
    
    testimg = readImgs(os.path.join(rspath,'test'), imgsize) / 255.0
    testy = nnPredict(testimg, response['parameters'])
    print (testy)


if __name__ == '__main__':
    main()      