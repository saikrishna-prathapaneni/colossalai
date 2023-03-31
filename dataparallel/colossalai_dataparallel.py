import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import colossalai
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader, MultiTimer
from colossalai.trainer import Trainer, hooks
from colossalai.nn.metric import Accuracy
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10
from colossalai.logging import get_dist_logger
from tqdm import tqdm



# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

if __name__=="__main__":

    colossalai.launch_from_torch(config='./config.py'
    )
    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=RandomDataset(input_size, data_size),batch_size=batch_size,shuffle=True)
    model = Model(input_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    engine, loader,test_loader,_ = colossalai.initialize(model,optimizer,criterion,rand_loader,test_loader)
    

    for data in test_loader:
        #engine.zero_grad()
        data=data.cuda()
        output = engine(data)
        print("Outside: input size", data.size(),
            "output_size", output.size())