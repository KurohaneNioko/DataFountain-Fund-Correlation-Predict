from csv2list import gen4submit
import pickle
import numpy
import torch
import torch.utils.data
import time
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class DataSet(torch.utils.data.Dataset):
    def __init__(self, keys, values):
        self.keys = torch.Tensor(keys).cuda()
        self.values = torch.Tensor(values).cuda()

    def __getitem__(self, index):
        indata = self.keys[index]
        outdata = self.values[index]
        return indata, outdata

    def __len__(self):
        return len(self.values)

class FundLoss(torch.nn.Module):
    def __init__(self):
        super(FundLoss, self).__init__()

    def forward(self, pred, truth):
        # MAE = (pred-truth).abs().mean()
        # TMAPE = ((pred-truth).abs() / (torch.full_like(truth, 1.5)-truth).abs()).mean
        return  torch.mean(torch.abs(pred-truth))+torch.mean(torch.abs(pred-truth) / torch.abs(torch.full_like(truth, 1.5).cuda()-truth))

class FundLSTM(torch.nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size, hidden_layers):
        super(FundLSTM, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            dropout=0.1,
            batch_first=True
        )
        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x.shape = (batch, time_step, input_size)
        # out.shape = (batch, time_step, output_size)
        # h_n (branch) (n_layers, batch, hidden_size)
        # h_c (main) (n_layers, batch, hidden_size)
        out, (h_n, h_c) = self.rnn(x, None)
        out = self.output(out[:, -61:, :])
        return out


def train(hs, hl, epmax, logfilename, train_loader, test_loader):
    '''
    with open('./train_data_' + str(fd1) + '_' + str(fd2) + '.p', 'rb') as f:
        data_list = pickle.load(f)
    with open('./train_label_' + str(fd1) + '_' + str(fd2) + '.p', 'rb') as f:
        label_list = pickle.load(f)
    dataset = DataSet(data_list, label_list)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=True)
    '''
    NN = FundLSTM(input_size=input_size, hidden_size=hs, hidden_layers=hl, output_size=output_size).cuda()
    lossfunc = FundLoss()
    optimizer = torch.optim.Adam(NN.parameters(), lr=lr, weight_decay=weight_decay)

    tr_ls = []
    te_ls = []
    T = 0.0
    for epoch in range(epmax):
        # print('current epoch = %d' % (epoch+1))
        train_loss = 0.0
        test_total_loss = 0.0
        train_batch_cnt, validation_batch_cnt = 0, 0
        # T = time.time()
        for key, value in train_loader:
            train_batch_cnt += 1
            key = torch.autograd.Variable(key)
            value = torch.autograd.Variable(value)
            optimizer.zero_grad()
            outputs = NN(key).squeeze()[:, -1]
            loss = lossfunc(outputs, value)
            loss.backward()
            optimizer.step()
            train_loss += loss
            # if i % 100 == 0:
            # print('current loss = %.5f' % loss.item())
        with torch.no_grad():
            for key, value in test_loader:
                validation_batch_cnt += 1
                outputs = NN(key).squeeze()[:, -1]
                loss = lossfunc(outputs, value)
                test_total_loss += loss
        tr_ls.append(train_loss.data / train_batch_cnt)
        te_ls.append(test_total_loss.data / validation_batch_cnt)
        # print(time.time()-T)
        if int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1 == epoch + 1 and numpy.min(te_ls) < 0.223:
            torch.save(NN.state_dict(), './weight_' + str(hs) + '_' + str(hl) + '_' + str(int(epoch + 1)) + '_' + str(
                numpy.min(te_ls)) + '.p')
        print(str(numpy.min(te_ls)) + ',' + str(int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1))

        if epoch > 0:
            epoches = range(1, epoch+2)
            plt.plot(epoches, tr_ls, label='Trainning Loss', color='blue')
            plt.plot(epoches, te_ls, label='Validation Loss', color='red')
            #plt.title('Loss')
            plt.xlabel('epoches')
            plt.ylabel('Loss')
            plt.legend()
            #plt.savefig(str(hs)+'_'+str(hl)+'.png')
            #plt.close('all')
            if (int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1 == epoch+1 and epoch>14) or epoch >=22:
                plt.show()
            else:
                plt.close('all')

    '''
    with open(logfilename, 'a+') as f:
        tr_ls = numpy.array(tr_ls)
        te_ls = numpy.array(te_ls)
        f.write(str(hs)+','+str(hl)+','+str(numpy.min(te_ls))+','+str(int((numpy.where(te_ls==numpy.min(te_ls)))[0])+1)+','+ 
                str(numpy.min(tr_ls))+','+str(int((numpy.where(tr_ls==numpy.min(tr_ls)))[0])+1))
        f.write('\n')
        f.close()
    '''
    epoches = range(1, epmax + 1)
    plt.plot(epoches, tr_ls, label='Trainning Loss', color='blue')
    plt.plot(epoches, te_ls, label='Validation Loss', color='red')
    plt.xlabel('epoches')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig(str(hs)+'_'+str(hl)+'.png')
    # plt.close('all')
    plt.show()
    print(str(hs) + ',' + str(hl) + ',' + str(numpy.min(te_ls)) + ',' + str(
        int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1) + ',' +
          str(numpy.min(tr_ls)) + ',' + str(int((numpy.where(tr_ls == numpy.min(tr_ls)))[0]) + 1))
    # torch.save(NN.state_dict(), './single_weight_'+str(hs)+'_'+str(hl)+'.p')
    # return NN


def train_all(hs, hl, epmax, logfilename, train_loader, test_loader):
    '''
    with open('./train_data_' + str(fd1) + '_' + str(fd2) + '.p', 'rb') as f:
        data_list = pickle.load(f)
    with open('./train_label_' + str(fd1) + '_' + str(fd2) + '.p', 'rb') as f:
        label_list = pickle.load(f)
    dataset = DataSet(data_list, label_list)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=True)
    '''
    NN = FundLSTM(input_size=input_size, hidden_size=hs, hidden_layers=hl, output_size=output_size).cuda()
    lossfunc = FundLoss() #torch.nn.L1Loss()
    optimizer = torch.optim.Adam(NN.parameters(), lr=lr, weight_decay=weight_decay)

    tr_ls = []
    te_ls = []
    T = 0.0
    for epoch in range(epmax):
        # print('current epoch = %d' % (epoch+1))
        train_loss = 0.0
        test_total_loss = 0.0
        train_batch_cnt, validation_batch_cnt = 0, 0
        # T = time.time()
        for key, value in train_loader:
            train_batch_cnt += 1
            key = torch.autograd.Variable(key)
            value = torch.autograd.Variable(value)
            optimizer.zero_grad()
            outputs = NN(key).squeeze().cuda()
            loss = lossfunc(outputs, value).cuda()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            train_loss += loss
            # if i % 100 == 0:
            # print('current loss = %.5f' % loss.item())
        with torch.no_grad():
            for key, value in test_loader:
                validation_batch_cnt += 1
                outputs = NN(key).squeeze()
                loss = lossfunc(outputs, value)
                test_total_loss += loss
        tr_ls.append(train_loss.data / train_batch_cnt)
        te_ls.append(test_total_loss.data / validation_batch_cnt)
        if int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1 == epoch+1:
            torch.save(NN.state_dict(), './weight_' + str(hs) + '_' + str(hl) + '_' + str(int(epoch+1)) + '.p')

        # if epoch > 0:
    epoches = range(1, epmax + 1)
    plt.plot(epoches, tr_ls, label='Trainning Loss', color='blue')
    plt.plot(epoches, te_ls, label='Validation Loss', color='red')
    plt.title('Loss')
    plt.xlabel('epoches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(hs) + '_' + str(hl) + '.png')
    plt.close('all')
    # plt.show()
    with open(logfilename, 'a+') as f:
        tr_ls = numpy.array(tr_ls)
        te_ls = numpy.array(te_ls)
        f.write(str(hs) + ',' + str(hl) + ',' + str(numpy.min(te_ls)) + ',' + str(
            int((numpy.where(te_ls == numpy.min(te_ls)))[0]) + 1) + ',' +
                str(numpy.min(tr_ls)) + ',' + str(int((numpy.where(tr_ls == numpy.min(tr_ls)))[0]) + 1))
        f.write('\n')
        f.close()
    torch.save(NN.state_dict(), './weight_' + str(hs) + '_' + str(hl) + '_'+ str(epmax) + '.p')

time_step = 61
batch_size = 512
lr = 1e-3
weight_decay = 1e-5
input_size = 38
output_size = 1

epochmax = 46
hidden_layers = [2,4,8,12]
hidden_size = [16,20,24,28]
h = []

if False:
    ''''''
    print(time.strftime("%H:%M:%S"), 'start', sep=' ')
    with open('./all_test_data.p', 'rb') as f1:
        d = pickle.load(f1)
        f1.close()
    with open('./all_test_label_1.p', 'rb') as f2:
        l = pickle.load(f2)
        f2.close()
    testset = DataSet(d, l)
    d, l = 0, 0
    print(time.strftime("%H:%M:%S"), 'build validate set  Over', sep=' ')
    with open('./all_data.p', 'rb') as f1:
        d = pickle.load(f1)
        f1.close()
    with open('./all_label_1.p', 'rb') as f2:
        l = pickle.load(f2)
        f2.close()
    print(time.strftime("%H:%M:%S"), 'read train  Over', sep=' ')
    trainset = DataSet(d, l)
    print(time.strftime("%H:%M:%S"), 'build train set  Over', sep=' ')
    ''''''
    d, l = 0,0

    # trainset = testset
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=True)
    '''
    print(time.strftime("%H:%M:%S"), '16, 2, 15')
    train_all(16, 2, 15, 'log.csv', train_loader, test_loader)
    print(time.strftime("%H:%M:%S"), 'over')
    print(time.strftime("%H:%M:%S"), '24, 2, 28')
    train_all(24, 2, 28, 'log.csv', train_loader, test_loader)
    print(time.strftime("%H:%M:%S"), 'over')
    
    for i in range(len(h)):
        e = h[i]
        print('hidden_size=', e[0], ' ', 'hidden_layers=', e[1], ' start', sep=' ')
        #T = time.time()
        train_all(e[0], e[1], 46, 'log.csv', train_loader, test_loader)
        print('hidden_size=', e[0], ' ', 'hidden_layers=', e[1], ' done', sep=' ')
        print(time.strftime("%H:%M:%S"))
    
    print(time.strftime("%H:%M:%S"), '36, 2, 46')
    train_all(36, 2, 46, 'log.csv', train_loader, test_loader)
    print(time.strftime("%H:%M:%S"), 'over')
    print(time.strftime("%H:%M:%S"), '32, 4, 86')
    train_all(32, 4, 86, 'log.csv', train_loader, test_loader)
    print(time.strftime("%H:%M:%S"), 'over')
    print(time.strftime("%H:%M:%S"), '24, 4, 86')
    train_all(24, 4, 86, 'log.csv', train_loader, test_loader)
    print(time.strftime("%H:%M:%S"), 'over')
    print(time.strftime("%H:%M:%S"), '36, 2, 86')
    '''
    while True:
        train(32, 2, 38, 'log.csv', train_loader, test_loader)
    print(time.strftime("%H:%M:%S"), 'over')

else:
    trainset, testset = 0,0
    hs = 32
    hl = 2
    ep = 12
    weight_path = 'weight_32_2_12_0.21314363' + '.p'
    RNN = FundLSTM(input_size=input_size, output_size=output_size,
                   hidden_size=hs, hidden_layers=hl).cuda()
    RNN.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage.cuda(0)))
    with open('result_' + str(hs) + '_' + str(hl) + '_'+ str(ep) + '.csv', 'w+') as f:
        f.write('ID,value\n')
        f.close()
    f = open('result_' + str(hs) + '_' + str(hl) + '_'+ str(ep) + '.csv', 'a+')
    for i in range(1, 200):
        for j in range(i + 1, 201):
            # d = gen_once(i, j)
            # RNN = train(hs=hidden_size[1], hl=hidden_layers[-1], epmax=4, datas=d)
            output_list = RNN(torch.autograd.Variable(torch.Tensor([gen4submit(i, j)]).cuda()))
            output_list = output_list.data.squeeze().cpu().numpy()
            f.write('Fund ' + str(i) + '-' + 'Fund ' + str(j) + ',' + str(output_list[-1]) + '\n')
            torch.cuda.empty_cache()
        print(i, 'done', sep=' ')