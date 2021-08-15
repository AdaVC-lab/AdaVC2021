from torch import nn
from components.custmized import ProbAttEncoder


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, layers=1):
        super(FeedForward, self).__init__()
        feed_forwards = []
        for i in range(layers):
            feed_forward = nn.Sequential(
                nn.Linear(in_dim if i is 0 else out_dim, out_dim, bias=bias),
                nn.ReLU()
            )
            feed_forwards.append(feed_forward)
        self.feed_forwards = nn.ModuleList(feed_forwards)

    def forward(self, x):
        for feed_forward in self.feed_forwards:
            x = feed_forward(x)
        return x


class TimbreConstraint(nn.Module):
    def __init__(self, h_params):
        super().__init__()
        factor = h_params['factor']
        dropout = h_params['dropout']
        n_heads = h_params['n_heads']
        d_ff = h_params['d_ff']
        activation = h_params['activation']

        t_hidden_dim = h_params['tim_hidden_dim']
        t_layers = h_params['tim_layers']
        tim_dim = h_params['tim_dim']

        self.encoder = ProbAttEncoder(factor, dropout, n_heads, t_hidden_dim, d_ff, activation, t_layers)
        self.linear_projection = nn.Linear(t_hidden_dim, tim_dim)

    def forward(self, hidden):
        hidden, _ = self.encoder(hidden)

        hidden = self.linear_projection(hidden)
        return hidden


class ProsodyConstraint(nn.Module):
    def __init__(self, h_params):
        super().__init__()
        factor = h_params['factor']
        dropout = h_params['dropout']
        n_heads = h_params['n_heads']
        d_ff = h_params['d_ff']
        activation = h_params['activation']

        p_hidden_dim = h_params['pro_hidden_dim']
        p_layers = h_params['pro_layers']
        pro_dim = h_params['pro_dim']

        self.encoder = ProbAttEncoder(factor, dropout, n_heads, p_hidden_dim, d_ff, activation, p_layers)
        self.feed_forward = FeedForward(p_hidden_dim, p_hidden_dim)
        self.output = nn.Linear(p_hidden_dim, pro_dim)

    def forward(self, hidden):
        hidden, _ = self.encoder(hidden)
        hidden = self.feed_forward(hidden)
        return self.output(hidden)


class ContentConstraint(nn.Module):
    def __init__(self, h_params):
        super().__init__()
        factor = h_params['factor']
        dropout = h_params['dropout']
        n_heads = h_params['n_heads']
        d_ff = h_params['d_ff']
        activation = h_params['activation']

        c_hidden_dim = h_params['cont_hidden_dim']
        c_layers = h_params['cont_layers']
        cont_dim = h_params['cont_dim']

        self.encoder = ProbAttEncoder(factor, dropout, n_heads, c_hidden_dim, d_ff, activation, c_layers)
        self.feed_forward = FeedForward(c_hidden_dim, c_hidden_dim)
        self.output = nn.Linear(c_hidden_dim, cont_dim)

    def forward(self, hidden):
        hidden, _ = self.encoder(hidden)
        hidden = self.feed_forward(hidden)
        return self.output(hidden)
