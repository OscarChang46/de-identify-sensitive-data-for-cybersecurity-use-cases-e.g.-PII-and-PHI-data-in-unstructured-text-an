from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

# sample data
def load_data():
    # passengers number of international airline , 1949-01 ~ 1960-12 per month
    seq_number = np.array(
        [112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
         118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
         114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
         162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
         209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
         272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
         302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
         315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
         318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
         348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
         362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
         342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
         417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
         432.], dtype=np.float32)
    # assert seq_number.shape == (144, )
    # plt.plot(seq_number)
    # plt.ion()
    # plt.pause(1)
    seq_number = seq_number[:, np.newaxis]

    # print(repr(seq))
    # 1949~1960, 12 years, 12*12==144 month
    seq_year = np.arange(12)
    seq_month = np.arange(12)
    seq_year_month = np.transpose(
        [np.repeat(seq_year, len(seq_month)),
         np.tile(seq_month, len(seq_year))],
    )  # Cartesian Product

    seq = np.concatenate((seq_number, seq_year_month), axis=1)

    # normalization
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    return seq


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size):
        super(LSTM, self).__init__()
        self.gate = nn.Linear(input_size + hidden_size, cell_size) 
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.output = nn.Linear(hidden_size, input_size)
        self.sigmoid =nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax()


    def forward(self, input):
        hidden = self.initHidden()
        cell = self.initCell()
        print(input.shape)
        print(hidden.shape)
        combined = torch.cat((input, hidden), 1)
        f_gate = self.gate(combined)
        f_gate = self.sigmoid(f_gate)
        i_gate = self.gate(combined)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.gate(combined)
        o_gate = self.sigmoid(o_gate)
        c_tilde = self.gate(combined)
        c_tilde = self.tanh(c_tilde)
        cell = torch.add(torch.mul(cell, f_gate) + torch.mul(c_tilde, i_gate))
        hidden = torch.mul(self.tanh(cell, o_gate))
        output = self.output(hidden)
        output = self.softmax(output)

        return output, hidden, cell
    
    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def initCell(self):
        return Variable(torch.zeros(1, self.cell_size))




class RegLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, cell_size):
        super(RegLSTM, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, cell_size)
        self.reg = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                        nn.Sigmoid(),
                        nn.Linear(hidden_size, output_size))
        self.h0 = self.lstm.initHidden()
        self.c0 = self.lstm.initCell()
    def forward(self, x):
        output = self.lstm(x, self.h0, self.c0)[0] # output, hidden_size, cell_size = self.lstm(x)
        seq_len, batch_size, hid_dim = output.shape
        output = output.view(-1, hid_dim)
        output = self.reg(output)
        output = output.view(seq_len, batch_size, -1)
        return output


def train_lstm():
    input_dim = 10
    hidden_dim = 5
    cell_dim = 5
    output_dim = 10

    data = load_data()

    data_x = data[:-1,:]
    data_y = data[+1:,0]
    assert(data_x.shape[1] == input_dim)

    train_size = int(len(data_x) * 0.75)

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




if __name__ == "__main__":
    import torch
    from torchvision.models import AlexNet
    from torchsummary import summary
    input_dim = 10
    hidden_dim = 5
    cell_dim = 5
    output_dim = 10
    seq_len = 108
    batch_size = 1
    input_dim = 3
    model = LSTM(input_dim, hidden_dim, cell_dim)
    # print(model)
    h0 = model.initHidden()
    c0 = model.initCell()
    input_size = (seq_len,1)
    summary(model, input_size, batch_size=1, device='cpu')
    

    '''
    from tensorboardX import SummaryWriter
    
    
    x=torch.rand(8,3,256,512)
    model=AlexNet()
    
    with SummaryWriter(comment='AlexNet') as w:
        w.add_graph(model, x) 
    '''
    
    # Visualize the graph: ensorboard --logdir Nov06_23-48-39_MacBook-Pro.localAlexNet