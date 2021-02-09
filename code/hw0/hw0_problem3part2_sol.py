# from the jupyter notebook provided by the class

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.datasets as ds
import pylab as plt

def load_mnist(datadir='./data_cache'):
    train_ds = ds.MNIST(root=datadir, train=True,
                           download=True, transform=None)
    test_ds = ds.MNIST(root=datadir, train=False,
                          download=True, transform=None)

    def to_xy(dataset):
        X = np.array(dataset.data) / 255.0  # [0, 1]
        Y = np.array(dataset.targets)
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    return X_tr, Y_tr, X_te, Y_te

X_tr, Y_tr, X_te, Y_te = load_mnist()

i = np.random.choice(len(X_tr))
plt.imshow(X_tr[i], cmap='gray')
plt.title(f'digit: {Y_tr[i]}')

print('original X_tr:', X_tr.shape)

# select 500 random examples
n = 500
I = np.random.choice(len(X_tr), n, replace=False)
X = X_tr[I]
Y = (Y_tr[I] % 2) * 2.0 - 1 # odd/even --> +1/-1
X = X.reshape(-1,  28*28) # flatten

print('reshaped X:', X.shape)
print('reshaped Y:', Y.shape)

# problem 3 part 2

def dLdbeta(X,Y,beta):
    if len(X.shape)==1:
        return np.multiply(np.dot(X.T,X), beta) - np.multiply(X.T,Y)
    else:
        return np.matmul(np.matmul(X.T,X), beta) - np.matmul(X.T,Y)

# this sgd uses all samples with minibatches of size 1. may want to add minibatches to this for better tradeoff between convergence and computational efficiency
def sgd(X, Y, learning_rate=0.01, n_epochs=50):

    beta = np.random.randn(784, )  # start with random beta

    epoch = 0

    while epoch < n_epochs:

        inds = np.arange(len(X))
        np.random.shuffle(inds) # shuffle the inds
        for i in inds:
            beta = beta-learning_rate*dLdbeta(X[i],Y[i],beta)
        epoch += 1
        learning_rate = learning_rate / 1.02 # gradually reducing the learning rate

    return beta

def gd(X, Y, learning_rate=0.0001, n_epochs=50):

    beta = np.random.randn(784, )  # start with random beta

    epoch = 0

    while epoch < n_epochs:

        beta = beta-learning_rate*dLdbeta(X,Y,beta)
        epoch += 1
        learning_rate = learning_rate / 1.02 # gradually reducing the learning rate

    return beta

# running SGD
print("SGD resutls")
beta_sgd = sgd(X,Y)

# double check:
train_error = np.linalg.norm(np.matmul(X,beta_sgd)-Y)**2/Y.shape[0]
print(f"train error: {train_error}")
print(f"train acc: {100*np.sum(np.round(np.matmul(X,beta_sgd))==Y)/Y.shape[0]}%")

# get the 'test error'
test_error = np.linalg.norm(np.matmul(X_te.reshape(-1,  28*28),beta_sgd)-Y_te)**2/Y_te.shape[0]
print(f"test error: {test_error}")
print(f"test acc: {100*np.sum(np.round(np.matmul(X_te.reshape(-1,  28*28),beta_sgd))==Y_te)/Y_te.shape[0]}%")



# running GD
print("GD resutls")
beta_gd = gd(X,Y)

# double check:
train_error = np.linalg.norm(np.matmul(X,beta_gd)-Y)**2/Y.shape[0]
print(f"train error: {train_error}")
print(f"train acc: {100*np.sum(np.round(np.matmul(X,beta_gd))==Y)/Y.shape[0]}%")

# get the 'test error'
test_error = np.linalg.norm(np.matmul(X_te.reshape(-1,  28*28),beta_gd)-Y_te)**2/Y_te.shape[0]
print(f"test error: {test_error}")
print(f"test acc: {100*np.sum(np.round(np.matmul(X_te.reshape(-1,  28*28),beta_gd))==Y_te)/Y_te.shape[0]}%")