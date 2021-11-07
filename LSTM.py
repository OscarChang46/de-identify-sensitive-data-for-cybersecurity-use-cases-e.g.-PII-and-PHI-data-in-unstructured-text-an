from torch.autograd import Variable
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size):
        super(LSTM, self).__init__()
        self.gate = nn.Linear(input_size + hidden_size, cell_size) 
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.output = nn.Linear(hidden_size, input_size)
        self.sigmoid =nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()


    def forward(self, input, hidden, cell):
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
        return Variable(torch.zeros(self.cell_size))




class RegLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, cell_size):
        super(self, RegLSTM).__init__()
        self.lstm = LSTM(input_size, hidden_size, cell_size)
        self.reg = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                        nn.Sigmoid(),
                        nn.Linear(hidden_size, output_size))

    def forward(self, x):
        output = self.lstm(x)[0] # output, hidden_size, cell_size = self.lstm(x)
        seq_len, batch_size, hid_dim = output.shape
        output = output.view(-1, hid_dim)
        output = self.reg(output)
        output = output.view()