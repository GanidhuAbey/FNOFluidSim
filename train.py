from neuralop.models import FNO
from neuralop.training import Trainer
from neuralop.losses import LpLoss, H1Loss

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader

save_path = './ckpt/'

class FluidH5Dataset(Dataset):
    def __init__(self, states, step_size=1, transform=None):
        self.states = states  # shape (N, T, 4, H, W)
        self.step_size = step_size
        self.transform = transform

        self.N, self.T, _, self.H, self.W = self.states.shape
        self.num_pairs = self.N * (self.T - step_size)

    def __len__(self):
        return self.num_pairs

    def compute_mean_std(self):
        # compute mean and std
        self.mean = torch.tensor(self.states.mean(axis=(0,1,3,4)), dtype=torch.float32).view(4,1,1)
        self.std  = torch.tensor(self.states.std(axis=(0,1,3,4)), dtype=torch.float32).view(4,1,1)


    def normalize(self, mean, std):
        self.states = (self.states - mean) / std

    # treat dataset set as long list of pairs (t0, t1), (t1, t2), ...
    # idx = 0 -> n = 0, t0 = 0, t1 = 1
    # idx = 1 -> n = 0, t0 = 1, t1 = 2
    # ...
    # idx = 199 -> n = 0, t0 = 198, t1 = 199
    # idx = 200 -> n = 1, t0=0, t1 = 1
    def __getitem__(self, idx):
        n = idx // (self.T - self.step_size)
        t = idx % (self.T - self.step_size)

        X = self.states[n, t]          # shape (4, H, W)
        Y = self.states[n, t + 1]      # shape (4, H, W)

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        if self.transform:
            X, Y = self.transform(X, Y)

        sample = {
            "x": X,   # input field
            "y": Y    # target field
        }
        return sample

def split_dataset(dataset_path, split_ratio):
    #dataset = FluidH5Dataset(dataset_path)
    with h5py.File(dataset_path, 'r') as f:
        dataset = f['states'][:]

    train_size = int(split_ratio[0] * dataset.shape[0])
    test_size = len(dataset) - train_size

    np.random.shuffle(dataset)

    train_ds = FluidH5Dataset(torch.from_numpy(dataset[:train_size]))
    test_ds = FluidH5Dataset(torch.from_numpy(dataset[train_size:]))

    print(dataset[train_size:].shape)

    return train_ds, test_ds

def train(train_data, test_data):
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)
    test_loaders = {'test': test_loader}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    operator = FNO(
        in_channels=4,      # u, v, p, rho
        out_channels=4,     # predict next-step u, v, p, rho
        n_modes=(32, 32),   # number of modes in fourier space
        hidden_channels=64, # internal channel width
    )

    trainer = Trainer(
        model=operator,
        n_epochs=20,
        verbose=True,
        device=device
    )

    # train the model
    optimizer = torch.optim.AdamW(operator.parameters(), lr=8e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # try different losses?
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss
    eval_losses = {"h1": h1loss, "l2": l2loss}

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
        save_best='test_l2'
    )

    return trainer


def main():
    dataset_path = 'fluid_training_data.h5'
    # split dataset
    split_ratio = [0.8, 0.2] # 80% train, 20% test data
    train_data, test_data = split_dataset(dataset_path, split_ratio)

    # preprocess dataset
    train_data.compute_mean_std();
    train_data.normalize(train_data.mean, train_data.std)
    test_data.normalize(train_data.mean, train_data.std)

    preprocesser = {
        'mean': train_data.mean,
        'std': train_data.std
    }

    torch.save(preprocesser, f'{save_path}preprocessor.pt')

    # train model
    model = train(train_data, test_data);

    # test model on one of the test datsets
    # idx = 0
    # t0, t1 = test_data[idx]
    # for i in range(0, 200):
    #     t1_hat = predict(model, t0)
    #     error += evaluate(t1_hat, t1)
    #     t0 = t1_hat
    #     t1 = 
    


if __name__ == "__main__":
    main()