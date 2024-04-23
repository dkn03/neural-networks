import torch
from torch.nn import RNN, LSTM, GRU


class SimpleRNN(torch.nn.Module):

    def __init__(self, input_size, num_layers, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_layer = RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        self.softmax = torch.nn.Softmax()
    
    def forward(self, x, hidden):
        x = torch.transpose(x, 0, 1)
        for i in range(len(x)):
            out, hn = self.rnn_layer(x[i], hidden)
        out = self.softmax(out)
        return out, hn

    def init_hidden(self):
        return torch.zeros(self.num_layers,  self.hidden_size)
    

class LSTMRNN(SimpleRNN):

    def __init__(self, input_size, num_layers, hidden_size):
        super().__init__(input_size, num_layers, hidden_size)
        self.rnn_layer = LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def init_hidden(self):
        return torch.zeros(self.input_size, self.num_layers, self.hidden_size)


class GRURNN(SimpleRNN):
    def __init__(self, input_size, num_layers, hidden_size):
        super().__init__(input_size, num_layers, hidden_size)
        self.rnn_layer = GRU(input_size, hidden_size, num_layers, batch_first=True)


class BidirectionalRNN(SimpleRNN):
    
    def __init__(self, input_size, num_layers, hidden_size):
        super().__init__(input_size, num_layers, hidden_size)
        self.rnn_layer = RNN(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)


    def init_hidden(self):
        return torch.zeros(self.num_layers * 2, self.hidden_size)
    

