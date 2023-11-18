import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime

import os
import sys
sys.path.append('./global_module/')
import network
import train
from generate_pic import aa_and_each_accuracy, sampling,load_dataset, generate_png, generate_iter
from Utils import fdssc_model, record, extract_samll_cubic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for Monte Carlo runs
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')

global Dataset  # UP,IN,KSC
dataset = input('Please input the name of Dataset(IP, UP, BS, SV, PC, KSC, HT):')
Dataset = dataset.upper()
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,TRAIN_VALIDATION_SPLIT = load_dataset(Dataset)

print(data_hsi.shape)
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = 10
PATCH_LENGTH = 4
# number of training samples per class
lr, num_epochs, batch_size = 0.00050, 150, 16

loss = torch.nn.CrossEntropyLoss()

img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE


KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

for index_iter in range(ITER):
    print('iter:', index_iter+1)
    net = network.TBTA_dense2net(BAND, CLASSES_NUM)

    optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=False) #, weight_decay=0.0001)
    # time_1 = int(time.time())
    np.random.seed(int(time.time()))

    train_indices, test_indices = sampling(TRAIN_VALIDATION_SPLIT, gt)
    _, total_indices = sampling(1, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')

    train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt)


    plt_path='./results/plts/'+Dataset+'/'+str(TRAIN_VALIDATION_SPLIT)+'/'
    if not os.path.exists(plt_path):
        os.makedirs(plt_path)

    tic1 = time.clock()
    train.train(net, train_iter, valida_iter, loss, optimizer, device, plt_path+str(index_iter+1)+'.png',epochs=num_epochs)
    toc1 = time.clock()

    pred_test_fdssc = []
    tic2 = time.clock()
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            net.eval()  # 评估模式, 这会关闭dropout
            # y_hat = net(X)
            pred_test_fdssc.extend(np.array(net(X).cpu().argmax(axis=1)))
    toc2 = time.clock()
    collections.Counter(pred_test_fdssc)
    gt_test = gt[test_indices] - 1


    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test[:-VAL_SIZE])
    confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test[:-VAL_SIZE])
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test[:-VAL_SIZE])


    net_save_path='./results/nets/' + Dataset + '_'+ str(TRAIN_VALIDATION_SPLIT)+'/'
    # net save path
    if not os.path.exists(net_save_path):
        os.makedirs(net_save_path)

    torch.save(net.state_dict(), net_save_path+str(index_iter+1)+ '_' + str(round(overall_acc_fdssc, 4)) + '.pt')
    KAPPA.append(kappa)
    OA.append(overall_acc_fdssc)
    AA.append(average_acc_fdssc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc_fdssc

print("--------" + net.name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     './results/records/' + net.name + day_str + '_' + Dataset + 'split：' + str(TRAIN_VALIDATION_SPLIT) + 'lr：' + str(lr) + '.txt')

generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices,TRAIN_VALIDATION_SPLIT)


