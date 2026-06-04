import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.functional.relu
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    def get_weight(self):
        flatten_weight = self.fc3.weight.reshape(-1)
        return flatten_weight

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool(self.relu(h))
        h = self.conv2(h)
        h = self.pool(self.relu(h))
        h = h.reshape(x.shape[0], -1)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        o = self.fc3(h)
        return o


def evaluate_dataloader(model, dataloader, show_progress=True):
    model.eval()

    correct_cnt, sample_cnt = 0, 0
    iterator = tqdm(dataloader) if show_progress else dataloader
    for img, label in iterator:
        pred = model(img)
        pred_label = pred.argmax(dim=1)

        correct_cnt += (pred_label == label).sum().item()
        sample_cnt += pred_label.shape[0]

        if show_progress:
            iterator.set_postfix(test_acc=correct_cnt / sample_cnt)

    return correct_cnt / sample_cnt if sample_cnt else 0.0


def random_index_generator(count):
    indices = np.arange(0, count)
    np.random.shuffle(indices)
    for idx in indices:
        yield idx


def compute_match_rate(model, X_wm, target_b, threshold=0.5):
    with torch.no_grad():
        # TODO: 结合模型特定层权重与水印向量X_wm，计算与target_b的一致性匹配率
        # NOTE: 可能用到的API: torch.sigmoid
        
        match_rate = 

    return match_rate  # match_rate为一个0～1之间的浮点数


class USPSDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

        if len(self.data.shape) == 2 and self.data.shape[1] == 256:
            self.data = self.data.reshape(-1, 16, 16)

        if self.data.dtype in (float, np.float32, np.float64):
            self.data = ((self.data + 1.0) / 2.0 * 255.0)
            self.data = np.clip(self.data, 0, 255).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label


def get_usps_loader(batch_size=128):
    
    transform = transforms.Compose(
        [
            # TODO: 定义USPS数据集的预处理方式
            # NOTE: 可能用到的API: transforms.Resize, transforms.ToTensor
        ]
    )

    train_mat = sio.loadmat("./USPS/USPStrainingdata.mat")
    test_mat = sio.loadmat("./USPS/USPStestingdata.mat")

    train_data = train_mat["traindata"]
    train_labels = train_mat["traintarg"].argmax(axis=1)

    test_data = test_mat["testdata"]
    test_labels = test_mat["testtarg"].argmax(axis=1)

    usps_train = USPSDataset(train_data, train_labels, transform=transform)
    usps_test = USPSDataset(test_data, test_labels, transform=transform)

    usps_train_loader = DataLoader(usps_train, batch_size=batch_size, shuffle=True)
    usps_test_loader = DataLoader(usps_test, batch_size=batch_size, shuffle=False)

    return usps_train_loader, usps_test_loader
