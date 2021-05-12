import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import math
import time

# 超参数
batchsize = 108
epochs = 1500
LR = 0.0001



def random_mini_batches(X, Y, mini_batch_size=64, seed=0):#一个划分mini-batch的函数
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k *
                                  mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k *
                                  mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,
                                  num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:,
                                  num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize_parameters():  # 权重字典
    W1 = torch.zeros((25, 12288)).cuda()
    W1.requires_grad=True
    nn.init.xavier_uniform_(W1, gain=nn.init.calculate_gain('relu'))
    b1 = torch.zeros((25, 1)).cuda()
    b1.requires_grad=True

    W2 = torch.zeros((12, 25)).cuda()
    W2.requires_grad=True
    nn.init.xavier_uniform_(W2, gain=nn.init.calculate_gain('relu'))
    b2 = torch.zeros((12, 1)).cuda()
    b2.requires_grad=True

    W3 = torch.zeros((6, 12)).cuda()
    W3.requires_grad=True
    nn.init.xavier_uniform_(W3)
    b3 = torch.zeros((6, 1)).cuda()
    b3.requires_grad=True

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward(X, parameters):#前向传播
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = W1@X + b1
    A1 = F.relu(Z1)
    Z2 = W2@A1 + b2
    A2 = F.relu(Z2)
    Z3 = W3@A2 + b3#无需softmax

    return Z3


def cost(Z3, Y):#计算交叉熵，包括softmax层
    crossentropyloss = nn.CrossEntropyLoss()
    return crossentropyloss(Z3.t(), Y)


def main():
    #开始时间
    start_time = time.process_time()
    # load data
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])# your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # your train set labels
    test_dataset = h5py.File('datasets/test_signs.h5',
                             "r")  # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  # your test set labels
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    Y_train = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    Y_test = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    # 预处理
    X_train_flatten = train_set_x_orig.reshape(
        train_set_x_orig.shape[0], -1).T  # 每一列就是一个样本
    X_test_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255
    # 训练
    (n_x, m) = X_train.shape
    costs = []
    seed = 3
    parameters = initialize_parameters()
    params=[]
    for key in parameters.keys():
        params.append(parameters[key])#优化器的params要求是参数的迭代器
        
    opt_Adam = optim.Adam(params, lr=LR, betas=(0.9, 0.99))
    for epoch in range(epochs):
        epoch_cost = 0
        num_batch = int(m / batchsize)
        seed = seed+1
        minibatches = random_mini_batches(X_train, Y_train, batchsize, seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            minibatch_X=torch.tensor(minibatch_X,dtype=torch.float32).cuda()#要指定数据类型，矩阵乘法要求类型相同
            minibatch_Y=torch.tensor(minibatch_Y.reshape(-1)).cuda()#计算cost时内置交叉熵函数要求minibatch_Y只有一个维度

            Z3 = forward(minibatch_X, parameters)
            batch_cost = cost(Z3, minibatch_Y)#前向传播计算损失

            opt_Adam.zero_grad()
            batch_cost.backward()#反向传播
            opt_Adam.step()

            epoch_cost += batch_cost.data/num_batch
        costs.append(epoch_cost)
        if epoch % 100 == 0:
            print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))
    #结束时间
    end_time = time.process_time()
    #计算时差
    print("GPU的执行时间 = " + str(end_time - start_time) + " 秒" )
    #测试
    X=torch.tensor(X_test,dtype=torch.float32).cuda()
    Y=torch.argmax(forward(X, parameters),dim=0).cpu().numpy()#取每一列的最大值索引
    Y_test=Y_test.reshape(-1)
    acc=np.sum(Y==Y_test)/Y_test.size*100
    print("train set accuracy："  , acc ,"%")



if __name__ == "__main__":
    main()


