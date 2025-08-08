import torch.nn as nn
import copy

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func)
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func)
        self.fc2 = get_clones(self.fc_h, self._layer_N)
        self.fc3 = nn.Sequential(
            init_(nn.Linear(hidden_size, output_dim)), active_func)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        x = self.fc3(x) + 1
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, action_dim = 10, hidden_size = 64, layer_N = 4, use_feature_normalization = False, use_orthogonal = True, use_ReLU = False):
        super(MLP, self).__init__()

        self._use_feature_normalization = use_feature_normalization
        self._use_orthogonal = use_orthogonal
        self._use_ReLU = use_ReLU
        self._layer_N = layer_N
        self.hidden_size = hidden_size

        obs_dim = input_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, action_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x