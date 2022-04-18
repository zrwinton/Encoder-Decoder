import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import scipy.io as scio
import h5py
import torch
import torch.nn as nn
import time
import numpy as np
import random
import torch.optim as optim
from torch.nn import init
from skimage.transform import resize
from PIL import Image

def dataset_preprocessing(pan_filename,
                          pan_key,
                          his_filename,
                          his_key,
                          w):

    pan = scio.loadmat(pan_filename)[pan_key]
    pan = pan.reshape(pan.shape[0], pan.shape[1], 1)
    his = scio.loadmat(his_filename)[his_key]
    target = np.zeros((pan.shape[0],pan.shape[1],his.shape[2]),dtype=np.float)
    for i in range(his.shape[2]):
        his_img = Image.fromarray(np.uint8(his[:, :, i]))
        target_img = his_img.resize((his_img.width * 12, his_img.height * 12),Image.ANTIALIAS)
        target[:,:,i] = np.asarray(target_img)
    pan_train_data = []
    his_train_data = []
    target_train_data = []

    for i in range(0, his.shape[0] - 1, w):
        for j in range(0, his.shape[1] - 1, w):
            tmp = his[i:i + w, j:j + w]
            tmp = tmp.transpose(2, 0, 1)
            his_train_data.append(tmp)
            tmp = pan[12 * i:12 * (i + w), 12 * j:12 * (j + w)]
            tmp = tmp.transpose(2, 0, 1)
            pan_train_data.append(tmp)
            tmp = target[i * 12:(i + w) * 12, j * 12:(j + w) * 12]
            tmp = tmp.transpose(2, 0, 1)
            target_train_data.append(tmp)

    pan_train_data = np.array(pan_train_data).astype(float)
    his_train_data = np.array(his_train_data).astype(float)
    target_train_data = np.array(target_train_data).astype(float)

    dict_train = {'pan':pan_train_data,
                  'his':his_train_data,
                  'target':target_train_data}

    pan_test_data = []
    his_test_data = []
    target_test_data = []

    for i in range(0, his.shape[0] - 1, w):
        for j in range(int(his.shape[1] * 0.8), his.shape[1] - 1, w):
            tmp = his[i:i + w, j:j + w]
            tmp = tmp.transpose(2, 0, 1)
            his_test_data.append(tmp)
            tmp = pan[12 * i:12 * (i + w), 12 * j:12 * (j + w)]
            tmp = tmp.transpose(2, 0, 1)
            pan_test_data.append(tmp)
            tmp = target[i * 12:(i + w) * 12, j * 12:(j + w) * 12]
            tmp = tmp.transpose(2, 0, 1)
            target_test_data.append(tmp)

    pan_test_data = np.array(pan_test_data).astype(float)
    his_test_data = np.array(his_test_data).astype(float)
    target_test_data = np.array(target_test_data).astype(float)

    dict_test = {'pan':pan_test_data,
                 'his':his_test_data,
                 'target':target_test_data}

    dict_Dataset = {'train':dict_train,
                    'test':dict_test}

    return dict_Dataset

class My_NN(nn.Module):
    def __init__(self):
        super(My_NN, self).__init__()

        self.his_up = nn.Sequential(
            nn.Upsample(scale_factor=12, mode='bicubic'),
        )
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=69,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.encoder_pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder_pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.encoder_pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )
        self.decoder_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
        )
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.decoder_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
        )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=68,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, his, pan):
        his_upsample = self.his_up(his)
        input = torch.cat((pan, his_upsample), 1)
        encod_conv1 = self.encoder_conv1(input)
        encod_pool1 = self.encoder_pool1(encod_conv1)
        encod_conv2 = self.encoder_conv2(encod_pool1)
        encod_pool2 = self.encoder_pool2(encod_conv2)
        encod_conv3 = self.encoder_conv3(encod_pool2)
        encod_pool3 = self.encoder_pool3(encod_conv3)
        decod_up1 = self.decoder_up1(encod_pool3)
        decod_up1 = decod_up1 + encod_conv3
        decod_conv1 = self.decoder_conv1(decod_up1)
        # decod_conv1 = decod_conv1 + encod_pool2
        decod_up2 = self.decoder_up2(decod_conv1)
        decod_up2 = decod_up2 + encod_conv2
        decod_conv2 = self.decoder_conv2(decod_up2)
        # decod_conv2 = decod_conv2 + encod_pool1
        decod_up3 = self.decoder_up3(decod_conv2)
        decod_up3 = decod_up3 + encod_conv1
        output = self.decoder_conv3(decod_up3)
        output = output + his_upsample
        return output

if __name__ == '__main__':

    dict_Dataset = dataset_preprocessing(pan_filename='dianchi_pan.mat',
                                         pan_key='dc_pan',
                                         his_filename='dianchi_hyper.mat',
                                         his_key='dc_hyper',
                                         w=20)

    net = My_NN()
    LR = 0.0005

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight.data)

    num_epochs = 1500
    batch_size = 16
    criterion = nn.MSELoss(reduce=True, size_average=True)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-6)

    losses = []
    time_start = time.time()
    train_loss_min = 99999
    test_loss_min = 99999
    for epoch in range(num_epochs):

        batch_loss = []

        for start in range(0, len(dict_Dataset['train']['pan']), batch_size):
            end = start + batch_size if start + batch_size < len(dict_Dataset['train']['pan']) else len(
                dict_Dataset['train']['pan'])
            pan_batch = torch.tensor(dict_Dataset['train']['pan'][start:end], dtype=torch.float)
            his_batch = torch.tensor(dict_Dataset['train']['his'][start:end], dtype=torch.float)
            target_batch = torch.tensor(dict_Dataset['train']['target'][start:end], dtype=torch.float)
            prediction = net(his_batch, pan_batch)
            loss = criterion(prediction, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.data.numpy())

        scheduler.step()

        if train_loss_min > np.mean(np.sqrt(batch_loss)):
            train_loss_min = np.mean(np.sqrt(batch_loss))
            torch.save(net, 'my_nn_ende_dc_1.pkl')

        time_end = time.time()
        print('epoch: ', epoch, ', loss: ', np.mean(np.sqrt(batch_loss)), ', time cost: ',
              time_end - time_start, ', loss min: ', train_loss_min)
        time_start = time.time()

    # w = 20
    #
    # pan = scio.loadmat('dianchi_pan.mat')['dc_pan']
    # pan = pan.reshape(1, pan.shape[0], pan.shape[1], 1)
    # pan = pan.astype(float)
    # his = scio.loadmat('dianchi_hyper.mat')['dc_hyper']
    # his = his.reshape(1, his.shape[0], his.shape[1], his.shape[2])
    # his = his.astype(float)
    #
    # output = np.zeros((his.shape[3], pan.shape[1], pan.shape[2]), dtype=np.float32)
    # output_div = np.zeros((his.shape[3], pan.shape[1], pan.shape[2]), dtype=np.float32)
    # print(output.shape)
    #
    # for i in range(his.shape[1] - w + 1):
    #     for j in range(his.shape[2] - w + 1):
    #         time_start = time.time()
    #         his_w = his[:, i:i + w, j:j + w, :]
    #         his_w = his_w.swapaxes(1, 3)
    #         his_w = his_w.swapaxes(2, 3)
    #         his_w = torch.from_numpy(his_w).float()
    #         pan_w = pan[:, 12 * i:12 * (i + w), 12 * j:12 * (j + w), :]
    #         pan_w = pan_w.swapaxes(1, 3)
    #         pan_w = pan_w.swapaxes(2, 3)
    #         pan_w = torch.from_numpy(pan_w).float()
    #         output[:, 12 * i:12 * (i + w), 12 * j:12 * (j + w)] = output[:, 12 * i:12 * (i + w),
    #                                                               12 * j:12 * (j + w)] + net(his_w, pan_w).data.numpy()
    #         output_div[:, 12 * i:12 * (i + w), 12 * j:12 * (j + w)] = output_div[:, 12 * i:12 * (i + w),
    #                                                                   12 * j:12 * (j + w)] + 1
    #         time_end = time.time()
    #         print(i, '  ', j, ' ', time_end - time_start)
    #
    # output = output.swapaxes(0, 2)
    # output = output.swapaxes(0, 1)
    # output_div = output_div.swapaxes(0, 2)
    # output_div = output_div.swapaxes(0, 1)
    #
    # for i in range(output.shape[0]):
    #     for j in range(output.shape[1]):
    #         for k in range(output.shape[2]):
    #             output[i, j, k] = output[i, j, k] / output_div[i, j, k]
    #
    # scio.savemat('dc_fusion_ende_1_1.mat', {'dc_fusion': output})
