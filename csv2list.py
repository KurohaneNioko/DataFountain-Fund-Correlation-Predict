import numpy
import csv
import random
import pickle
import concurrent.futures
import time

parentdir = './CCF/fund/'
days = 61
lim = 539   #最大 小于539
'''
import numpy.random as nprd
def gen():
    r = nprd.random()
    if nprd.random() > 0.85:
        return (r*-0.15)
    else:
        while r < 0.4:
            r = nprd.random()
        return (r)
def outputcsv():
    with open(parentdir + 'submit_exmaple.csv', 'r') as f:
        reader = csv.reader(f)
        with open(parentdir + '/result.csv', 'w+', newline ='') as fw:
            writer = csv.writer(fw)
            for row in reader:
                # print (row)
                if row[0] == 'ID':
                    writer.writerow(row)
                else:
                    row[1] = str(gen())
                    writer.writerow(row)
'''

def outputcsv(filename):
    with open(parentdir + 'submit_exmaple.csv', 'r') as f:
        reader1 = csv.reader(f)
        with open('./'+filename[:-4] + 'result.csv', 'w+', newline ='') as fw:
            writer = csv.writer(fw)
            with open('./'+filename, 'r') as fr:
                reader2 = csv.reader(fr)
                for row1 in reader1:
                    if row1[0] == 'ID':
                        writer.writerow(row1)
                    else:
                        break
                for row1, row2 in reader1, reader2:
                    row1[1] = str(row2[2])
                    writer.writerow(row1)

def str2floatMULT10(x):
    return 10.0*float(x)


def read_index(arg = 'train'):      # 一行是一天的指数
    if arg=='train' or arg=='test' or arg=='all':
        root = parentdir + arg + '_index_return.csv'
    else:
        return None
    indexs = []
    with open(root, 'r', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == '':
                continue
            else:
                indexs.append(list(map(str2floatMULT10, row[1:])))
    return numpy.transpose(indexs)

def read_fun_return(fd1, fd2, arg = 'train'):
    """ 一行是 基金fd小的涨跌幅+benchmark+基金fd大的涨跌幅+benchmark """
    if (arg=='train' or arg=='test' or arg=='all') and fd1!=fd2 and 1<=fd1<=200 and 1<=fd2<=200:
        root1 = parentdir + arg +'_fund_return.csv'
        root2 = parentdir + arg +'_fund_benchmark_return.csv'
    else:
        return None
    s, l = min(fd1, fd2), max(fd1, fd2)
    fund = []
    with open(root1, 'r', errors='ignore') as f1:
        with open(root2, 'r', errors='ignore') as f2:
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)
            a = list(reader1)
            b = list(reader2)
            fund.append(list(map(str2floatMULT10, a[s][1:])))
            fund.append(list(map(str2floatMULT10, b[s][1:])))
            fund.append(list(map(str2floatMULT10, a[l][1:])))
            fund.append(list(map(str2floatMULT10, b[l][1:])))
    return numpy.transpose(fund)

def get_data(fd1, fd2, type='train'):
    """ 一行是 34个指数 + 基金fd小的涨跌幅+benchmark+基金fd大的涨跌幅+benchmark """
    return numpy.hstack((read_index(type), read_fun_return(fd1, fd2, type))).tolist()

def label_lineNo(fd1, fd2):
    assert (1<=fd1<=200 and 1<=fd2<=200 )
    f1, f2 = min(fd1, fd2), max(fd1, fd2)
    return int((199+(200-(f1-1)))*(f1-1)/2+f2-f1)

def get_label(fd1, fd2, type='train'):
    """ 400 (139 539) 个相关系数 fd1和fd2的 """
    if (type=='train' or type=='test' or type=='all') and fd1!=fd2 and 1<=fd1<=200 and 1<=fd2<=200:
        root = parentdir + type + '_correlation.csv'
    else:
        return None
    with open(root, 'r', errors='ignore') as f:
        reader = csv.reader(f)
        a = list(map(float, list(reader)[label_lineNo(fd1, fd2)][1:]))
        return a


'''
def get_2funds(fd1, fd2, type='train'):
    y = get_label(fd1, fd2, type)
    return numpy.column_stack((get_data(fd1, fd2, type), y)).tolist(), y

def data4use(fd1, fd2, type='train'):
    key = []
    value = []
    k = get_data(fd1, fd2, type)
    v = get_label(fd1, fd2, type)
    end = 1
    end += 400-days if type == 'train' else 139-days if type=='test' else 0
    assert end>1
    for i in range(end):
        #a = k[i:i+days]
        #b = v[i:i+days-61]+([0.0]*(61))
        key.append(numpy.column_stack( (k[i:i+days], (v[i:i+days-61]+([0.0]*(61)))) ).tolist())
        value.append(v[i+days-61:i+days])
    return key, value
'''


def gen4submit(fd1, fd2):
    k = get_data(fd1, fd2, 'test')
    # v = get_label(fd1, fd2, 'test')
    # i =
    #key = numpy.column_stack((k[i:i + days], (v[i:i + days - 61] + ([0.0] * 61)))).tolist()
    # value = v[i + days - 61:i + days]
    return k[200-days : 200] #ey# , value


def gen_from_all(fd):
    """从所有数据中生成，对每对基金对都来一趟，在539之前生成"""
    fd1 = fd[0]
    fd2 = fd[1]
    datas = []
    labels = []
    all_data = get_data(fd1, fd2, 'all')
    all_label = get_label(fd1, fd2, 'all')
    continuous_count = divmod(lim, days)[0] - 1     #连续的time_step大小的数据块
    cross_count = continuous_count - 1      #覆盖交界处
    start = 0
    for i in range(continuous_count):
        j = i+1
        datas.append(all_data[start:start+days])
        labels.append(all_label[start:start+days])
        if j<continuous_count:
            ss = random.randint(start+5, start+days-5)   #5 is magic
            datas.append(all_data[ss:ss + days])
            labels.append(all_label[ss:ss + days])
        start += days
    with open('./each/train_'+str(fd1)+'_'+str(fd2)+'.p', 'w+b') as f:
        pickle.dump((datas,labels), f)
    print(fd, time.strftime("%H:%M:%S"))
    # return datas, labels

def gen4test(fd):
    fd1 = fd[0]
    fd2 = fd[1]
    datas = []
    labels = []
    test_data = get_data(fd1, fd2, 'test')
    test_label = get_label(fd1, fd2, 'test')

    datas.append(test_data[0:days])
    labels.append(test_label[0:days])

    datas.append(test_data[-days:])
    labels.append(test_label[-days:])

    ss = random.randint(30, 60)
    datas.append(test_data[ss:ss+days])
    labels.append(test_label[ss:ss+days])

    with open('./each/test_' + str(fd1) + '_' + str(fd2) + '.p', 'w+b') as f:
        pickle.dump((datas, labels), f)
    print(fd, time.strftime("%H:%M:%S"))

if __name__ == '__main__':
    params = []
    for i in range(14, 200):
        for j in range(i+1, 201):
            params.append((i, j))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map(gen_from_all, params)
        executor.map(gen4test, params)

    td = []
    tl = []
    for e in params:
        fd1 = e[0]
        fd2 = e[1]
        with open('./each/test_' + str(fd1) + '_' + str(fd2) + '.p', 'rb') as f:
            d, l = pickle.load(f)
        td += d
        tl += l
    with open('./all_test_data.p', 'w+b') as f:
        pickle.dump(td, f)
    with open('./all_test_label.p', 'w+b') as f:
        pickle.dump(tl, f)


