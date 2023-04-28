from core import compute_ee, Channels, determine_kappa
from core import SINRnet, SINRnetLocal
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import argparse
record = False
try:
    from tensorboardX import SummaryWriter
except:
    record = False

params = {
          "channel_data": r"data/channels_7users_training.pt",
          "results_path": "results/",
          "num_UE": 7,
          "feature_dim": 20,
          "num_iters": 6,
          "inefficiency": 4.,
          "band_width": 180e3,
          "p_max": 1,
          "scaler": 1,
          "device": "cpu",
          "pC": 1,
          "batch_size": int(1024 / 2),
          "epoch": 8000001,
          "num_rsamples": 1000,
          "lr": 5e-7,
          "objective_entropy": -10,
          "saving_frequency": 1000,
          "global_optimum": True,
          }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmax")
    parser.add_argument("--record")
    args = parser.parse_args()
    if args.record is not None:
        record = args.record == "True"
    if args.pmax is not None:
        params["p_max"] = 10 ** float(args.pmax)
        if params["p_max"] < 1:
            params["scaler"] = params["p_max"]
    now = datetime.now()
    if record:
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        folder_name = dt_string + "_pmax={pmax:1.2f}".format(pmax=params["p_max"])
        Path("results/" + folder_name).mkdir(parents=True, exist_ok=True)
        params["results_path"] = params["results_path"] + folder_name + "/"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params["device"] = device
    data_set = Channels(params, device)
    train_loader = DataLoader(dataset=data_set, batch_size=params["batch_size"], shuffle=True)
    if params["global_optimum"]:
        model = SINRnet(params, device)
    else:
        model = SINRnetLocal(params, device)

    optimizer = optim.Adam(model.parameters(), params['lr'])
    if record:
        writer = SummaryWriter(logdir=params["results_path"])
    model.train()
    losses = list()
    entropies = list()
    for _ in range(data_set.__len__()):
        entropies.append(list())
    previous_kappa = torch.zeros(data_set.__len__()).to(device)
    kappa = torch.zeros(data_set.__len__()).to(device)

    for epoch in range(params['epoch']):
        losses_current_epoch = list()
        entropies_current_epoch = list()
        for channel_indices, channels in train_loader:
            optimizer.zero_grad()
            if params["global_optimum"]:
                power, entropy, a, b = model(channels)
                power *= params["scaler"]
                a *= params["scaler"]
                b *= params["scaler"]
                for channel_idx in channel_indices:
                    kappa[channel_idx] = determine_kappa(entropies[channel_idx][-50:], previous_kappa[channel_idx])
                previous_kappa = kappa
                loss, ee, penalty = compute_ee(channels, power, a, b, entropy, kappa[channel_indices], params, device)
            else:
                power = model(channels)
                entropy = 0
                loss, ee, penalty = compute_ee(channels, power, power, power,
                                               entropy, kappa[channel_indices], params, device)
            loss.backward()

            optimizer.step()

            losses_current_epoch.append(ee.item())
            if params["global_optimum"]:
                entropies_current_epoch.append(entropy.mean().cpu().detach().numpy())
                for idx_in_batch, channel_idx in enumerate(channel_indices):
                    entropies[channel_idx].append(entropy[idx_in_batch].cpu().detach().numpy())
                    while len(entropies[channel_idx]) > 50:
                        entropies[channel_idx].pop(0)
        losses.append(-np.mean(losses_current_epoch))
        if params["global_optimum"]:
            print("Epoch {epoch}, EE: {ee}, Entropy: {entropy}.".format(epoch=epoch,
                                                                        ee=ee,
                                                                        entropy=np.mean(entropies_current_epoch)))
        else:
            print("Epoch {epoch}, EE: {ee}.".format(epoch=epoch, ee=ee))
        if record and epoch % 100 == 0:
            writer.add_scalar("Training/EE", ee, epoch)
            writer.add_scalar("Training/kappa", previous_kappa.mean(), epoch)
            if params["global_optimum"]:
                writer.add_scalar("Training/Entropy", entropy.mean(), epoch)
                writer.add_scalar("Training/penalty", penalty, epoch)

        if record and epoch % params['saving_frequency'] == 0:
            torch.save(model.state_dict(), params['results_path'] + 'model_{epoch}'.format(epoch=epoch))
            if record:
                writer.flush()
