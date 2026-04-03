import os

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

import pickle
from matplotlib import pyplot as plt


########## IMPLEMENT THE CODE BELOW, COMMENT OUT IRRELEVENT CODE IF NEEDED ##########
##### Model Definition #####
# TODO: Week 5, Task 1
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # TODO：在这里定义模型的结构。主要依赖于nn.Conv2d和nn.MaxPool2d、nn.functional.relu、nn.Linear
        # LeNet-5的结构参考课程PPT，注意卷积层和全连接层的输入输出维度

    def forward(self, x):
        # TODO：在这里定义前向传播过程。注意这里输入的x形状是[Batch, 1, 28, 28]
        # 中间注意需要使用reshape/view把卷积层的输出展平为全连接层的输入
        # LeNet-5的结构参考课程PPT，按照PPT中的层顺序实现前向传播过程
        
        return o


##### Model Evaluation #####
# TODO: Week 5, Task 2
def evaluate(imgs, labels, model):
    # TODO：用model预测imgs，并得到预测标签pred_label
    model.eval()
    pred = 
    pred_label = 

    # TODO：计算预测标签与真实标签的匹配数目
    correct_cnt = 
    
    print(f'match rate: {correct_cnt/labels.shape[0]}')
    return pred_label

# TODO: Week 7, Task 1
def evaluate_dataloader(dataloader, model):
    model.eval()
    
    correct_cnt, sample_cnt = 0, 0

    t = tqdm(dataloader)
    for img, label in t:
        # TODO: Predict label for img, update correct_cnt, sample_cnt

        t.set_postfix(test_acc=correct_cnt/sample_cnt)

##### Adversarial Attacks #####
# TODO: Week 5, Task 2
def fgsm(imgs, epsilon, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()
    adv_xs.requires_grad = True

    # TODO：模型前向传播，计算loss，然后loss反传

    # TODO：得到输入的梯度、生成对抗样本
    # 公式：adv_xs = imgs + epsilon * sign(grad)，其中grad是输入的梯度，sign()是符号函数

    # TODO：对扰动做截断，保证对抗样本的像素值在合理域内
    # 使用函数：torch.clamp(input, min, max)可以对输入张量的元素进行截断，使其值在指定的范围内

    model.train()

    return adv_xs.detach()


# TODO: Week 6, Task 1
def pgd(imgs, epsilon, alpha, iter, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()

    for i in range(iter):
        # Forward and compute loss, then backward

        # Retrieve grad and generate adversarial example, note to detach

        # Clip perturbation
        pass
    
    model.train()

    return adv_xs.detach()


# TODO: Week 6, Task 2
def fgsm_target(imgs, epsilon, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()
    adv_xs.requires_grad = True

    # Forward and compute loss, then backward

    # Retrieve grad and generate adversarial example, note to detach
    # Note to compute TARGETED loss and the sign of the perturbation

    # Clip perturbation

    model.train()

    return adv_xs.detach()

# TODO: Week 6, Task 2
def pgd_target(imgs, epsilon, alpha, iter, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()

    for i in range(iter):
        adv_xs.requires_grad = True
        
        # Forward and compute loss, then backward

        # Retrieve grad and generate adversarial example, note to detach
        # Note to compute TARGETED loss and the sign of the perturbation

        # Clip perturbation
    
    model.train()

    return adv_xs.detach()

# TODO: Week 6, Bonus
def nes(imgs, epsilon, model, labels, sigma, n):
    """
    labels: ground truth labels
    sigma: search variance
    n: number of samples used for estimation for each img
    """
    model.eval()

    adv_xs = imgs.reshape(-1, 28 * 28).float()

    grad = torch.zeros_like(adv_xs)
    # TODO: Estimate gradient for each sample adv_x in adv_xs

    adv_xs = adv_xs.detach() - epsilon * grad.sign()
    adv_xs = torch.clamp(adv_xs, min=0., max=1.)

    model.train()

    return adv_xs.detach()


########## NO NEED TO MODIFY CODE BELOW ##########
##### Data Loader #####
def load_mnist(batch_size):
    if not os.path.exists('data/'):
        os.mkdir('data/')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = torchvision.datasets.MNIST(root='data/', transform=transform, train=True, download=True)
    test_set = torchvision.datasets.MNIST(root='data/', transform=transform, train=False, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


##### Visualization #####
def visualize_benign(imgs, labels):
    fig = plt.figure(figsize=(8, 7))
    for idx, (img, label) in enumerate(zip(imgs, labels)):
        ax = fig.add_subplot(4, 5, idx + 1)
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f'label: {label.item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def visualize_adv(imgs, true_labels, pred_labels):
    fig = plt.figure(figsize=(8, 8))
    for idx, (img, true_label, pred_label) in enumerate(zip(imgs, true_labels, pred_labels)):
        ax = fig.add_subplot(4, 5, idx + 1)
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f'true label: {true_label.item()}\npred label: {pred_label.item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def visualize_target_adv(imgs, target_labels, pred_labels):
    fig = plt.figure(figsize=(8, 8))
    for idx, (img, true_label, pred_label) in enumerate(zip(imgs, target_labels, pred_labels)):
        ax = fig.add_subplot(4, 5, idx + 1)
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f'target label: {true_label.item()}\npred label: {pred_label.item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
