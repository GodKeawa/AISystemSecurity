import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from copy import deepcopy

def get_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = model.feature_extractor(inputs)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            all_images.append(inputs.cpu())
            
    return torch.cat(all_features), torch.cat(all_labels), torch.cat(all_images)

def select_targets_and_bases(train_features, train_labels, train_images, 
                             test_features, test_labels, test_images, num_classes=10):
    target_indices = []
    base_indices = []
    
    for c in range(num_classes):
        c_test_indices = (test_labels == c).nonzero(as_tuple=True)[0]
        selected_targets = c_test_indices[torch.randperm(len(c_test_indices))[:10]]
        
        t = (c + 1) % num_classes
        
        # Find all train samples of target class t
        t_train_indices = (train_labels == t).nonzero(as_tuple=True)[0]
        
        for target_idx in selected_targets:
            target_feat = test_features[target_idx]
            
            # Mask out already selected base indices
            valid_mask = torch.ones(len(t_train_indices), dtype=torch.bool)
            for i, idx in enumerate(t_train_indices):
                if idx.item() in base_indices:
                    valid_mask[i] = False
                    
            valid_t_train_indices = t_train_indices[valid_mask]
            valid_t_train_features = train_features[valid_t_train_indices]
            
            distances = torch.norm(valid_t_train_features - target_feat.unsqueeze(0), dim=1)
            closest_idx = valid_t_train_indices[torch.argmin(distances)]
            
            target_indices.append(target_idx.item())
            base_indices.append(closest_idx.item())
            
    return target_indices, base_indices

def optimize_poison_sample(model, base_img, target_feat, device, max_iter=1000, lr=2.0, beta=0.25):
    """
    Implements Forward-Backward Splitting from Algorithm 1 of the Poison Frogs paper.
    Scaled for CIFAR-10 and normalized images.
    """
    model.eval()
    
    b = base_img.to(device)
    target_feat = target_feat.to(device)
    
    # Initialize x0 = b
    x = b.clone().detach()
    
    # Scale beta based on dimension ratio (reference code uses this exact scaling logic)
    # feature dim = 512, image pixels = 3*32*32 = 3072
    # scaled_beta = beta * (512 / 3072)^2 = beta * 0.0277
    scaled_beta = beta * (512.0 / 3072.0)**2
    
    for i in range(max_iter):
        x.requires_grad_(True)
        
        poison_feat = model.feature_extractor(x.unsqueeze(0)).squeeze()
        
        # Forward step: Lp(x) = ||f(x) - f(t)||^2
        L_p = torch.norm(poison_feat - target_feat) ** 2
        
        # Calculate gradient \nabla_x Lp(x_{i-1})
        grad_x = torch.autograd.grad(L_p, x)[0]
        
        with torch.no_grad():
            # Forward step: \hat{x}_i = x_{i-1} - \lambda \nabla_x L_p(x_{i-1})
            x_hat = x - lr * grad_x
            
            # Backward step: x_i = (\hat{x}_i + \lambda \beta b)/(1 + \beta \lambda)
            x = (x_hat + lr * scaled_beta * b) / (1 + scaled_beta * lr)
        
    return x.detach().cpu()

def create_poisoned_dataset(train_dataset, target_indices, base_indices, test_images, test_features, model, device):
    poisoned_dataset = deepcopy(train_dataset)
    
    poisons = []
    bases = []
    targets = []
    
    print("Optimizing poison samples...")
    for target_idx, base_idx in tqdm(zip(target_indices, base_indices), total=len(target_indices)):
        base_img, _ = train_dataset[base_idx]
        target_img = test_images[target_idx]
        target_feat = test_features[target_idx]
        
        poison_img = optimize_poison_sample(model, base_img, target_feat, device)
        
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2471, 0.2435, 0.2616]).view(3, 1, 1)
        
        poison_img_unnorm = (poison_img * std) + mean
        poison_img_uint8 = (poison_img_unnorm.clamp(0, 1) * 255).byte().numpy().transpose(1, 2, 0)
        
        poisoned_dataset.data[base_idx] = poison_img_uint8
        
        poisons.append(poison_img)
        bases.append(base_img)
        targets.append(target_img)
        
    return poisoned_dataset, poisons, bases, targets

def finetune_model(model, train_loader, epochs, lr, device):
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    return model
