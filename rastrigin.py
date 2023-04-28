import torch
import numpy as np
import torch.optim as optim
from torch.distributions.uniform import Uniform
from core import determine_kappa_v6
import matplotlib.pyplot as plt


def ackley(x):
    d = x.shape[-1]
    return (-20 * torch.exp(-0.2 * torch.sqrt(1 / d * torch.sum(x ** 2)))
            - torch.exp(1 / d * torch.sum(torch.cos(2 * np.pi * x))) + 20 + np.exp(1))


def rastrigin(x):
    n = x.shape[-1]
    a = 10
    return a * n + torch.sum(x ** 2 - a * torch.cos(2 * np.pi * x))


if __name__ == "__main__":
    separated_dimensions = False
    dim = 10
    a = (torch.rand(dim) - 4).requires_grad_()
    b = (torch.rand(dim) + 6).requires_grad_()

    if separated_dimensions:
        kappa = torch.zeros((1, dim))
    else:
        kappa = torch.zeros(1)
    optimizer = optim.SGD([a, b], 1e-5)
    losses = list()
    entropies = list()
    objs = list()
    kappas = list()
    if separated_dimensions:
        entropy_long_memory = np.zeros((1, dim))
        entropy_short_memory = np.zeros((1, dim))
    else:
        entropy_long_memory = np.zeros(1)
        entropy_short_memory = np.zeros(1)

    fun_a_long_memory = 0
    fun_a_short_memory = 0
    fun_b_long_memory = 0
    fun_b_short_memory = 0

    obj_memory = 0
    cum_obj_long = 0

    discount_long = 0.95
    discount_short = 0.5
    fun_a = rastrigin(a)
    fun_b = rastrigin(b)
    min_criterium = 0

    # a0 = np.zeros((2000000, dim))
    # b0 = np.zeros((2000000, dim))
    # kappa0 = np.zeros((2000000, dim))

    for iter in range(2000000):
        fun_a_previous = fun_a
        fun_b_previous = fun_b
        optimizer.zero_grad()
        bb = torch.maximum(a + 1e-6, b)
        fun_a = rastrigin(a)
        fun_b = rastrigin(bb)
        pi_distribution = Uniform(a, bb)
        x = pi_distribution.rsample()
        if separated_dimensions:
            entropy = (bb - a)
        else:
            entropy = (bb - a).mean()
        entropies.append(entropy.mean().item())

        obj = rastrigin(x)
        objs.append(obj.item())

        # entropy_long_memory, entropy_short_memory, \
        # obj_long_memory, obj_short_memory, \
        # kappa = determine_kappa_v5(entropy, entropy_long_memory, entropy_short_memory,
        #                            obj, fun_a_long_memory, fun_a_short_memory,
        #                            kappa, discount_long, discount_short)

        (kappa, obj_memory, cum_obj_long,
         entropy_long_memory,
         entropy_short_memory) = determine_kappa_v6(kappa, fun_a, fun_b,
                                                    obj, obj_memory, cum_obj_long,
                                                    entropy_long_memory, entropy_short_memory, entropy,
                                                    discount_long, discount_short, iter)

        kappas.append(kappa.mean().item())
        # a0[iter, :] = a.cpu().detach().numpy()
        # b0[iter, :] = bb.cpu().detach().numpy()
        # if separated_dimensions:
        #     kappa0[iter, :] = kappa.cpu().detach().numpy()
        # else:
        #     kappa0[iter] = kappa.cpu().detach().numpy()
        loss = obj + (torch.tensor(kappa) * entropy).sum()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if iter % 1000 == 0:
            print("Iter: {iter}, obj: {obj}, entropy: {ent}, kappa: {kappa}.".
                  format(iter=iter, obj=obj,
                         ent=entropy.mean(), kappa=kappa.mean()))
        if obj == 0 and entropy.mean() < 1.5e-6:
            break
    # np.save("results/rastrigin_losses.npy", losses)
    # np.save("results/rastrigin_obj.npy", objs)
    # np.save("results/rastrigin_kappa.npy", kappas)
    # np.save("results/rastrigin_entropy.npy", entropies)
