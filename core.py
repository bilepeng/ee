import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.utils.data import Dataset


class SINRnet(nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.act_limit = 1
        self.params = params
        # state, hidden layer, action sizes
        self.s_size = params['num_UE']
        self.h_size = params['num_UE']
        self.a_size = params['num_UE']
        # define layers
        self.conv1a = nn.Conv2d(1, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv2a = nn.Conv2d(1 + params["feature_dim"] * 4, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv3a = nn.Conv2d(1 + params["feature_dim"] * 4, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv4a = nn.Conv2d(1 + params["feature_dim"] * 4, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv5a = nn.Conv2d(1 + params["feature_dim"] * 4, 1, (1, 1)).to(device)
        self.conv1b = nn.Conv2d(1, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv2b = nn.Conv2d(1 + params["feature_dim"] * 4, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv3b = nn.Conv2d(1 + params["feature_dim"] * 4, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv4b = nn.Conv2d(1 + params["feature_dim"] * 4, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv5b = nn.Conv2d(1 + params["feature_dim"] * 4, 1, (1, 1)).to(device)
        # Extraction matrix
        self.feature_dim = params["feature_dim"]
        self.ex = (torch.ones((params["num_UE"], params["num_UE"])) - torch.eye(params["num_UE"])).to(device)
        self.n_cat1 = params["num_UE"] - 1
        self.n_cat2 = params["num_UE"] - 1
        self.n_cat3 = (params["num_UE"] - 1) ** 2

    def forward(self, channel):
        def postprocess_layer(channel_, conv_output):
            features_c0 = conv_output[:, : self.feature_dim, :, :]
            features_c1 = self.ex @ conv_output[:, self.feature_dim: (self.feature_dim * 2), :, :] / self.n_cat1
            features_c2 = conv_output[:, (self.feature_dim * 2): (self.feature_dim * 3), :, :] @ self.ex / self.n_cat2
            features_c3 = (self.ex @ conv_output[:, (self.feature_dim * 3): (self.feature_dim * 4), :, :]
                           @ self.ex / self.n_cat3)
            return torch.cat((channel_, features_c0, features_c1, features_c2, features_c3), 1)
        if channel.dim() == 2:
            channel = channel[None, None, :, :]
        elif channel.dim() == 3:
            channel = channel[:, None, :]
        a = F.relu(self.conv1a(channel))
        a = postprocess_layer(channel, a)
        a = F.relu(self.conv2a(a))
        a = postprocess_layer(channel, a)
        a = F.relu(self.conv3a(a))
        a = postprocess_layer(channel, a)
        a = F.relu(self.conv4a(a))
        a = postprocess_layer(channel, a)
        a = self.conv5a(a)
        a = torch.diagonal(a, dim1=2, dim2=3)
        a = torch.squeeze(a)
        a = a - 2
        bma = F.relu(self.conv1b(channel))
        bma = postprocess_layer(channel, bma)
        bma = F.relu(self.conv2b(bma))
        bma = postprocess_layer(channel, bma)
        bma = F.relu(self.conv3b(bma))
        bma = postprocess_layer(channel, bma)
        bma = F.relu(self.conv4b(bma))
        bma = postprocess_layer(channel, bma)
        bma = self.conv5b(bma)
        bma = torch.diagonal(bma, dim1=2, dim2=3)
        bma = torch.squeeze(bma)
        bma = bma + 4
        bma = torch.maximum(bma, torch.tensor(0.00000001))
        pi_distribution = Uniform(a, a + bma)
        entropy = pi_distribution.entropy().sum(dim=1)
        action = pi_distribution.rsample()
        return action[:, None, :], entropy, a, a + bma


class SINRnetLocal(nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.act_limit = 1
        self.params = params
        # state, hidden layer, action sizes
        self.s_size = params['num_UE']
        self.h_size = params['num_UE']
        self.a_size = params['num_UE']
        # define layers
        self.conv1a = nn.Conv2d(1, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv2a = nn.Conv2d(1 + params["feature_dim"] * 4, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv3a = nn.Conv2d(1 + params["feature_dim"] * 4, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv4a = nn.Conv2d(1 + params["feature_dim"] * 4, params["feature_dim"] * 4, (1, 1)).to(device)
        self.conv5a = nn.Conv2d(1 + params["feature_dim"] * 4, 1, (1, 1)).to(device)
        # Extraction matrix
        self.feature_dim = params["feature_dim"]
        self.ex = (torch.ones((params["num_UE"], params["num_UE"])) - torch.eye(params["num_UE"])).to(device)
        self.n_cat1 = params["num_UE"] - 1
        self.n_cat2 = params["num_UE"] - 1
        self.n_cat3 = (params["num_UE"] - 1) ** 2

    def forward(self, channel):
        def postprocess_layer(channel_, conv_output):
            features_c0 = conv_output[:, : self.feature_dim, :, :]
            features_c1 = self.ex @ conv_output[:, self.feature_dim: (self.feature_dim * 2), :, :] / self.n_cat1
            features_c2 = conv_output[:, (self.feature_dim * 2): (self.feature_dim * 3), :, :] @ self.ex / self.n_cat2
            features_c3 = (self.ex @ conv_output[:, (self.feature_dim * 3): (self.feature_dim * 4), :, :]
                           @ self.ex / self.n_cat3)
            return torch.cat((channel_, features_c0, features_c1, features_c2, features_c3), 1)
        if channel.dim() == 2:
            channel = channel[None, None, :, :]
        elif channel.dim() == 3:
            channel = channel[:, None, :]
        a = F.relu(self.conv1a(channel))
        a = postprocess_layer(channel, a)
        a = F.relu(self.conv2a(a))
        a = postprocess_layer(channel, a)
        a = F.relu(self.conv3a(a))
        a = postprocess_layer(channel, a)
        a = F.relu(self.conv4a(a))
        a = postprocess_layer(channel, a)
        a = self.conv5a(a)
        a = torch.diagonal(a, dim1=2, dim2=3)
        a = torch.squeeze(a)
        return a[:, None, :]


class Channels(Dataset):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.num_UE = params['num_UE']
        self.all_channels = torch.load(params["channel_data"]).to(device)

    def __getitem__(self, item):
        return item, self.all_channels[item, :, :]

    def __len__(self):
        return self.all_channels.shape[0]


def compute_ee(channel, power, a, b, entropy, kappa, params, device="cpu"):
    if len(power.shape) == 2:
        power = power[:, None, :]
        a = a[:, None, :]
        b = b[:, None, :]
    penalty = torch.sum(torch.maximum(torch.zeros(1).to(device), b - params["p_max"]), dim=1)
    penalty += torch.sum(torch.maximum(torch.zeros(1).to(device), -a), dim=1)
    penalty /= params["scaler"]
    channel = 10 ** channel
    power = torch.clip(power, 0, 1e3)
    rss = channel * power
    signal = torch.diagonal(rss, dim1=1, dim2=2)
    interference = torch.sum(rss - torch.diag_embed(signal), dim=2)
    rate = torch.log2(1 + signal / (1 + interference))
    total_power = params["inefficiency"] * power[:, 0, :] + params["pC"]
    ee = params["band_width"] * 1e-6 * torch.sum(rate/total_power, dim=1)
    return -torch.mean(ee - kappa * entropy - penalty * 100), torch.mean(ee), penalty.mean()


def determine_kappa(entropies, previous_kappa):
    lr_kappa = 1e-6
    if len(entropies) < 10:
        return previous_kappa
    elif entropies[-1] >= np.mean(entropies[:-1]):
        return previous_kappa + lr_kappa
    else:
        return F.relu(previous_kappa - lr_kappa / 2)


def determine_kappa_v6(kappa,
                       fun_a, fun_b,
                       obj, cum_obj_short, cum_obj_long,
                       cum_support_long, cum_support_short, support,
                       discount_long, discount_short,
                       iter):
    lr = 0.2
    fun_a = fun_a.item()
    fun_b = fun_b.item()
    obj = obj.item()
    cum_obj_short *= discount_short
    cum_obj_short += obj
    cum_obj_long *= discount_long
    cum_obj_long += obj
    support = support.detach().cpu().numpy()

    cum_support_long *= discount_long
    cum_support_long += support
    cum_support_short *= discount_short
    cum_support_short += support

    if iter > 100:
        baseline = cum_obj_short * (1 - discount_short)
        criterium = np.maximum(fun_a - baseline, fun_b - baseline)
        if criterium < 0:
            if cum_obj_short * (1 - discount_short) > cum_obj_long * (1 - discount_long):
                kappa += lr
        else:
            kappa = np.maximum(kappa - lr, 0)
    return kappa, cum_obj_short, cum_obj_long, cum_support_long, cum_support_short

