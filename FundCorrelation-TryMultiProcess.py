
# coding: utf-8

# In[1]:


from csv2list import gen4submit
import time
import pickle
import numpy
import torch
import torch.utils.data
import time
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class DataSet(torch.utils.data.Dataset):
    def __init__(self, keys, values):
        self.keys = torch.Tensor(keys).cuda().share_memory_()
        self.values = torch.Tensor(values).cuda().share_memory_()

    def __getitem__(self, index):
        indata = self.keys[index]
        outdata = self.values[index]
        return indata, outdata

    def __len__(self):
        return len(self.values)


class FundLSTM(torch.nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size, hidden_layers):
        super(FundLSTM, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
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


# In[3]:


print(time.strftime("%H:%M:%S"), 'start', sep=' ')
with open('./all_data.p', 'rb') as f1:
    d = pickle.load(f1)
    f1.close()
with open('./all_label.p', 'rb') as f2:
    l = pickle.load(f2)
    f2.close()
print(time.strftime("%H:%M:%S"), 'read train  Over', sep=' ')
trainset = DataSet(d, l)
d, l = 0, 0
print(time.strftime("%H:%M:%S"), 'build train set  Over', sep=' ')
with open('./all_test_data.p', 'rb') as f1:
    d = pickle.load(f1)
    f1.close()
with open('./all_test_label.p', 'rb') as f2:
    l = pickle.load(f2)
    f2.close()
testset = DataSet(d, l)
d, l = 0, 0
print(time.strftime("%H:%M:%S"), 'build validate set  Over', sep=' ')
#trainset = testset


# In[8]:


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
    lossfunc = torch.nn.L1Loss()
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
            outputs = NN(key).squeeze()
            loss = lossfunc(outputs, value)
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
        # print(time.time()-T)
        #if epoch > 0:
    epoches = range(1, epmax+1)
    plt.plot(epoches, tr_ls, label='Trainning Loss', color='blue')
    plt.plot(epoches, te_ls, label='Validation Loss', color='red')
    plt.title('Loss')
    plt.xlabel('epoches')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(hs)+'_'+str(hl)+'.png')
    plt.close('all')
    #plt.show()
    with open(logfilename, 'a+') as f:
        tr_ls = numpy.array(tr_ls)
        te_ls = numpy.array(te_ls)
        f.write(str(hs)+','+str(hl)+','+str(numpy.min(te_ls))+','+str(int((numpy.where(te_ls==numpy.min(te_ls)))[0])+1)+','+ 
                str(numpy.min(tr_ls))+','+str(int((numpy.where(tr_ls==numpy.min(tr_ls)))[0])+1))
        f.write('\n')
        f.close()
    torch.save(NN.state_dict(), './weight_'+str(hs)+'_'+str(hl)+'.p')
    #return NN


# In[9]:


def test(hs, hl, fd1, fd2):
    weight_path = './weight_'+str(hs)+'_'+str(hl)+'.p'
    net = FundLSTM(input_size=input_size, output_size=output_size, hidden_size=hs, hidden_layers=hl).cuda()
    net.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage.cuda(0)))
    with open('test_data_' + str(fd1) + '_' + str(fd2) + '.p', 'rb') as file_in:
        read_data = pickle.load(file_in)
    output_list = net(torch.autograd.Variable(torch.Tensor(read_data).cuda()))
    output_list = output_list.data.squeeze().cpu().numpy()
    print(output_list)
    with open('test_label_' + str(fd1) + '_' + str(fd2) + '.p', 'rb') as f:
        real = numpy.array(pickle.load(f))
    print((numpy.array(output_list) -  real).mean())


# In[10]:


time_step = 61
batch_size = 512
lr = 1e-3
weight_decay = 1e-5
input_size = 38
output_size = 1

epochmax = 46
hidden_layers = [2,4,8,12,16]
hidden_size = [16,20,24,28]
h = [(16,2), (16,4), (16,8), (16,12), (20,2), (20,4), (20,8), (20,12), (20,16),
     (24,2), (24,4), (24,8), (24,12), (24,16), (28,2), (28,4), (28,8), (28,12), (28,16)]
train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=True)


# In[11]:


torch.set_num_threads(6)
for i in range(len(h)):
    e = h[i]
    if(e[0]<=16 and e[1]<8):
        continue
    print('hidden_size=',e[0], ' ', 'hidden_layers=', e[1], ' start', sep=' ')
    T = time.time()
    train(e[0], e[1], 46, 'log.csv', train_loader, test_loader)
    print('hidden_size=',e[0], ' ', 'hidden_layers=', e[1], ' done', sep=' ')
    print(time.time() - T)
exit(6)

# In[ ]:

'''
weight_path = './weight_'+str(hs)+'_'+str(hl)+'.p'
RNN = FundLSTM(input_size=input_size, output_size=output_size,
               hidden_size=hs, hidden_layers=hl).cuda()
RNN.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage.cuda(0)))
with open('result.csv', 'w+') as f:
    f.write('ID,value\n')
    f.close()
for i in range(1, 200):
    for j in range(i+1, 201):
        # d = gen_once(i, j)
        # RNN = train(hs=hidden_size[1], hl=hidden_layers[-1], epmax=4, datas=d)
        output_list = RNN(torch.autograd.Variable(torch.Tensor([gen4submit(i, j)]).cuda()))
        output_list = output_list.data.squeeze().cpu().numpy()
        # print(output_list)
        # os.system('del *.p')
        with open('result.txt', 'a+') as f:
            f.write('Fund '+str(i)+'-'+'Fund '+str(j) + ',' + str(output_list[-1])+'\n')
            f.close()
        torch.cuda.empty_cache()
        print(i, j, 'done', sep=' ')

'''