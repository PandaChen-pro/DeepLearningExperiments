import torch
import torch.nn as nn
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from dataset_loader import load_dataset
from trainer import train_model
import torchvision.models as models
import wandb
import os
from model import ViT
import timm

from torchvision.models import ViT_B_16_Weights, vit_b_16

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset_path = "/home/code/experiment/deeplearningExperiment/experiment1/data/train"
    val_dataset_path = "/home/code/experiment/deeplearningExperiment/experiment1/data/val"
    num_classes = 10
    num_epochs = 300
    warmup_ratio = 0.2
    batch_size = 512
    wandb.login()
    wandb.init(project='ViT-CIFAR10-original-0410', name='ViT-CIFAR10-original-Experiment-0410')

    train_loader, val_loader = load_dataset(train_dataset_path, val_dataset_path, batch_size=batch_size, is_train_shuffle=True, is_val_shuffle=False,num_workers=4)
    # 不使用预训练模型，使用小尺寸的模型
    model = ViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.3,
        attn_drop_rate=0.3
    )

    # 使用预训练模型
    # model = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
    # model.heads = nn.Sequential(
    #     nn.Linear(model.heads.head.in_features, num_classes)
    # )
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # for param in model.encoder.layers[-1].parameters():
    #     param.requires_grad = True
    # for param in model.heads.parameters():
    #     param.requires_grad = True
    

    # 加载上一个checkpoint继续训练，/home/code/experiment/modal/resnext/checkpoints/Cancer_Val_Epoch23_Acc79.69.pth
    # checkpoint_path = "/data/coding/deep_learning_experiments/experiment2/checkpoints/ViT_Original_Val_Epoch97_Acc74.67.pth"
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    # if 'optimizer_state_dict' in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)  
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_steps,        
        lr_min=1e-5,                  
        warmup_t=warmup_steps,       
        warmup_lr_init=1e-6,       
        t_in_epochs=False,          
        cycle_limit=1,             
        warmup_prefix=True          
    )

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,num_epochs)