from main import params
from core import SINRnet, compute_ee, Channels
import torch
from torch.utils.data import DataLoader


if __name__ == "__main__":
    params["channel_data"] = "data/channels_7users_validation.pt"
    model = SINRnet(params)
    model.load_state_dict(torch.load("results/model_28000"))  # Trained model
    data_set = Channels(params)
    train_loader = DataLoader(dataset=data_set, batch_size=600, shuffle=True)
    for channel_indices, channels in train_loader:
        power, entropy, a, b = model.forward(channels)
        loss, ee, penalty = compute_ee(channels, power, a, b, entropy, 0, params)
        print(ee.mean())
        pass