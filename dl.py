import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt

"""
Github: Yonv1943 Zen4 Jia1 hao2
https://github.com/Yonv1943/DL_RL_Zoo/blob/master/RNN

The source of training data 
https://github.com/L1aoXingyu/
code-of-learn-deep-learning-with-pytorch/blob/master/
chapter5_RNN/time-series/lstm-time-series.ipynb
"""


def run_train_gru():
    inp_dim = 3
    out_dim = 1
    batch_size = 12 * 4

    '''load data'''
    data = load_data()
    data_x = data[:-1, :]
    data_y = data[+1:, 0]
    assert data_x.shape[1] == inp_dim

    train_size = int(len(data_x) * 0.75)

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    train_x = train_x.reshape((train_size, inp_dim))
    train_y = train_y.reshape((train_size, out_dim))

    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegGRU(inp_dim, out_dim, mod_dim=12, mid_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    '''train'''
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

    batch_var_x = list()
    batch_var_y = list()

    for i in range(batch_size):
        j = train_size - i
        batch_var_x.append(var_x[j:])
        batch_var_y.append(var_y[j:])

    from torch.nn.utils.rnn import pad_sequence
    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    with torch.no_grad():
        weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    for e in range(256):
        out = net(batch_var_x)

        # loss = criterion(out, batch_var_y)
        loss = (out - batch_var_y) ** 2 * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 100 == 0:
            print('Epoch: {}, Loss: {:.5f}'.format(e, loss.item()))

    '''eval'''
    net = net.eval()

    test_x = data_x.copy()
    test_x[train_size:, 0] = 0
    test_x = test_x[:, np.newaxis, :]
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
    for i in range(train_size, len(data) - 2):
        test_y = net(test_x[:i])
        test_x[i + 1, 0, 0] = test_y[-1]
    pred_y = test_x[1:, 0, 0]
    pred_y = pred_y.cpu().data.numpy()

    diff_y = pred_y[train_size:] - data_y[train_size:-1]
    l1_loss = np.mean(np.abs(diff_y))
    l2_loss = np.mean(diff_y ** 2)
    print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))
    plt.plot(pred_y, 'r', label='pred')
    plt.plot(data_y, 'b', label='real')
    plt.legend(loc='best')
    plt.pause(4)


def run_train_lstm(dataset, p=6):
    inp_dim = 2
    out_dim = 1
    mid_dim = 8
    mid_layers = 1
    batch_size = 12 * 4
    mod_dir = '.'

    # '''load data'''
    # data = load_data(dataset)
    # data_x = data[:-1, :]
    # data_y = data[+1:, 0]
    # # assert data_x.shape[1] == inp_dim

    # train_size = int(len(data_x) * 0.8)

    # train_x = data_x[:train_size]
    # train_y = data_y[:train_size]
    # train_x = train_x.reshape((train_size, inp_dim))
    # train_y = train_y.reshape((train_size, out_dim))
    
    '''load data'''
    data, seq_mean, seq_std = load_data(dataset)
    data_x = data[:-1, :]
    data_y = data[+1:, 0]

    # Use all data for training
    train_x = data_x
    train_y = data_y

    train_size = len(data_x)  # update train_size

    train_x = train_x.reshape((train_size, inp_dim))
    train_y = train_y.reshape((train_size, out_dim))

    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    '''train'''
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

    batch_var_x = list()
    batch_var_y = list()

    for i in range(batch_size):
        j = train_size - i
        batch_var_x.append(var_x[j:])
        batch_var_y.append(var_y[j:])

    from torch.nn.utils.rnn import pad_sequence
    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    with torch.no_grad():
        weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Training Start")
    for e in range(64*12):
        out = net(batch_var_x)
    
        # loss = criterion(out, batch_var_y)
        loss = (out - batch_var_y) ** 2 * weights
        loss = loss.mean()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if e % 64 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))
    torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))
    print("Save in:", '{}/net.pth'.format(mod_dir))

    # '''eval'''
    # net.load_state_dict(torch.load('{}/net.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
    # net = net.eval()

    # test_x = data_x.copy()
    # test_x[train_size:, 0] = 0
    # test_x = test_x[:, np.newaxis, :]
    # test_x = torch.tensor(test_x, dtype=torch.float32, device=device)

    '''prediction'''
    net = net.eval()

    # Use the last 'p' data points from training data to predict future
    last_p_data = train_x[-p:]
    last_p_data = last_p_data[np.newaxis, :]  # Add one dimension for batch
    last_p_data = torch.tensor(last_p_data, dtype=torch.float32, device=device)
    
    # Initialize hidden and cell states
    h0 = torch.zeros(mid_layers, p, mid_dim, device=device)
    c0 = torch.zeros(mid_layers, p, mid_dim, device=device)
    hc = (h0, c0)
    # Predict 'p' steps into the future
    future_y = np.empty((0, out_dim))
    # Use the last state of the LSTM from training as initial state for prediction
    next_y, hc = net.output_y_hc(last_p_data, hc)
    next_y = next_y.cpu().data.numpy()
    future_y = np.append(future_y, next_y[-1])  # Take the last prediction as output

    # '''prediction'''
    # net = net.eval()

    # # Assuming you want to predict 'p' steps into the future
    # future_data = np.zeros((p, inp_dim))  # initialize future data
    # future_data = future_data[:, np.newaxis, :]
    # future_data = torch.tensor(future_data, dtype=torch.float32, device=device)

    # # Initialize hidden and cell states
    # h0 = torch.zeros(mid_layers, 1, mid_dim, device=device)  # batch size set to 1
    # c0 = torch.zeros(mid_layers, 1, mid_dim, device=device)  # batch size set to 1
    # hc = (h0, c0)

    # # Use the last state of the LSTM from training as initial state for prediction
    # future_y, hc = net.output_y_hc(future_data, hc)
    # future_y = future_y.cpu().data.numpy()
    
    # Convert future_y to 1D for simplicity
    future_y = future_y.squeeze()
    
    
    # '''simple way but no elegant'''
    # for i in range(train_size, len(data) - 2):
    #     test_y = net(test_x[:i])
    #     test_x[i, 0, 0] = test_y[-1]

    # '''elegant way but slightly complicated'''
    # eval_size = 1
    # zero_ten = torch.zeros((mid_layers, eval_size, mid_dim), dtype=torch.float32, device=device)
    # test_y, hc = net.output_y_hc(test_x[:train_size], (zero_ten, zero_ten))
    # test_x[train_size + 1, 0, 0] = test_y[-1]
    # for i in range(train_size + 1, len(data) - 2):
    #     test_y, hc = net.output_y_hc(test_x[i:i + 1], hc)
    #     test_x[i + 1, 0, 0] = test_y[-1]
    # pred_y = test_x[1:, 0, 0]
    # pred_y = pred_y.cpu().data.numpy()

    # diff_y = pred_y[train_size:] - data_y[train_size:-1]
    # l1_loss = np.mean(np.abs(diff_y))
    # l2_loss = np.mean(diff_y ** 2)
    # print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))

    # plt.plot(pred_y, 'r', label='pred')
    # plt.plot(data_y, 'b', label='real', alpha=0.3)
    # # plt.plot([train_size, train_size], [-1, 2], color='k', label='train | pred')
    # plt.legend(loc='best')
    
    
    # # Convert future_y to 1D for simplicity
    # future_y = future_y.squeeze()
    

    # Concatenate the known data (data_y) and the predicted future data
    data_y = data_y * seq_std + seq_mean
    future_y = future_y * seq_std + seq_mean
    all_data = np.concatenate((data_y, future_y))
    
    # If you want to calculate loss, you need actual future values.
    # Since we don't have them, we can't calculate and print L1 and L2 loss here.

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(all_data, 'r', label='prediction')
    plt.plot(data_y, 'b', label='real', alpha=1)
    plt.legend(loc='best')
    plt.title('Count of Requests')
    plt.xlabel('Minutes')
    plt.ylabel('Counts')
    
    plt.savefig('lstm_reg.png')
    # plt.pause(4)
    
    return future_y


def run_origin():
    inp_dim = 2
    out_dim = 1
    mod_dir = '.'

    '''load data'''
    data = load_data()  # axis1: number, year, month
    data_x = np.concatenate((data[:-2, 0:1], data[+1:-1, 0:1]), axis=1)
    data_y = data[2:, 0]

    train_size = int(len(data_x) * 0.75)
    train_x = data_x[:train_size]
    train_y = data_y[:train_size]

    train_x = train_x.reshape((-1, 1, inp_dim))
    train_y = train_y.reshape((-1, 1, out_dim))

    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim=4, mid_layers=2).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    '''train'''
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)
    print('var_x.size():', var_x.size())
    print('var_y.size():', var_y.size())

    for e in range(512):
        out = net(var_x)
        loss = criterion(out, var_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e + 1) % 100 == 0:  # 每 100 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

    torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))

    '''eval'''
    # net.load_state_dict(torch.load('{}/net.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
    net = net.eval()  # 转换成测试模式

    """
    inappropriate way of seq prediction: 
    use all real data to predict the number of next month
    """
    test_x = data_x.reshape((-1, 1, inp_dim))
    var_data = torch.tensor(test_x, dtype=torch.float32, device=device)
    eval_y = net(var_data)  # 测试集的预测结果
    pred_y = eval_y.view(-1).cpu().data.numpy()

    plt.plot(pred_y[1:], 'r', label='pred inappr', alpha=0.3)
    plt.plot(data_y, 'b', label='real', alpha=0.3)
    plt.plot([train_size, train_size], [-1, 2], label='train | pred')

    """
    appropriate way of seq prediction: 
    use real+pred data to predict the number of next 3 years.
    """
    test_x = data_x.reshape((-1, 1, inp_dim))
    test_x[train_size:] = 0  # delete the data of next 3 years.
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
    for i in range(train_size, len(data) - 2):
        test_y = net(test_x[:i])
        test_x[i, 0, 0] = test_x[i - 1, 0, 1]
        test_x[i, 0, 1] = test_y[-1, 0]
    pred_y = test_x.cpu().data.numpy()
    pred_y = pred_y[:, 0, 0]
    plt.plot(pred_y[2:], 'g', label='pred appr')

    plt.legend(loc='best')
    plt.savefig('lstm_origin.png')
    plt.pause(4)


class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    """
    PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:
    Examples::
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    def output_y_hc(self, x, hc):
        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc


class RegGRU(nn.Module):
    def __init__(self, inp_dim, out_dim, mod_dim, mid_layers):
        super(RegGRU, self).__init__()

        self.rnn = nn.GRU(inp_dim, mod_dim, mid_layers)
        self.reg = nn.Linear(mod_dim, out_dim)

    def forward(self, x):
        x, h = self.rnn(x)  # (seq, batch, hidden)

        seq_len, batch_size, hid_dim = x.shape
        x = x.view(-1, hid_dim)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x

    def output_y_h(self, x, h):
        y, h = self.rnn(x, h)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, h


def load_data(dataset):
    # Splitting data and time into separate arrays
    seq_data = dataset[:, 0]
    seq_time = dataset[:, 1]

    # Reshaping to 2D arrays
    seq_data = seq_data.reshape(-1, 1)
    seq_time = seq_time.reshape(-1, 1)
    
    # Concatenating data and time side by side
    seq = np.concatenate((seq_data, seq_time), axis=1)

    # Normalization
    # Calculate mean and standard deviation
    seq_mean = seq.mean(axis=0)
    seq_std = seq.std(axis=0)
    seq = (seq - seq_mean) / seq_std
    return seq, seq_mean[0], seq_std[0]


if __name__ == '__main__':
    run_train_lstm()
    # run_train_gru()
    # run_origin()