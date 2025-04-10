import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from model import ResNext50
from dataset_loader import load_dataset
import wandb
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm
import torchvision.models as models
import torch.optim as optim
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, 
                val_acc_threshold=0.98, patience=10, min_delta=0.001):
    """
    训练模型并实现早停机制
    
    参数:
    - model: 要训练的模型
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - scheduler: 学习率调度器
    - num_epochs: 最大训练轮数
    - val_acc_threshold: 验证准确率阈值，达到此值时提前终止训练
    - patience: 早停耐心值，验证性能不提升超过此轮数时提前终止
    - min_delta: 最小变化阈值，性能提升需超过此值才算有效提升
    """
    best_val_acc = 0
    best_model_path = None
    val_acc_threshold_percent = val_acc_threshold * 100
    
    # 早停相关变量
    counter = 0  # 计数器：验证性能未提升的连续轮数
    best_val_loss = float('inf')  # 记录最佳验证损失
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step_update(epoch * len(train_loader) + i)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 30 == 0:
                batch_acc = 100 * correct / total
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "batch": epoch * len(train_loader) + i,
                    "train_loss": loss.item(),
                    "train_acc": batch_acc,
                    "learning_rate": current_lr
                })
                tqdm.write(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}, learning_rate: {current_lr:.6f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        print('Starting validation...')
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        
        # 记录到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 检查是否需要保存模型
        os.makedirs("./checkpoints", exist_ok=True)
        if val_acc > best_val_acc:
            if best_model_path is not None and os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                    print(f'删除旧的最佳模型: {best_model_path}')
                except Exception as e:
                    print(f'删除旧模型失败: {e}')
            best_val_acc = val_acc
            new_model_path = f"./checkpoints/ViT_Original_0410_Val_Epoch{epoch+1}_Acc{val_acc:.2f}.pth"
            
            # 保存完整的模型状态，包括优化器状态
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, new_model_path)
            
            print(f'保存模型到 {new_model_path}')
            best_model_path = new_model_path
            
            # 重置早停计数器
            counter = 0
        else:
            # 如果验证准确率没有提高，增加计数器
            counter += 1
            print(f'验证准确率未提升，早停计数: {counter}/{patience}')
        
        # 早停条件检查 - 基于验证损失
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            # 不重置计数器，因为我们主要关注准确率
        
        # 检查是否满足早停条件
        # if counter >= patience:
        #     print(f'验证准确率连续 {patience} 个epoch未提升，触发早停')
        #     break
            
        # 检查是否达到准确率阈值
        if val_acc >= val_acc_threshold_percent:
            print(f'验证集准确率 ({val_acc:.2f}%) 已达到阈值 ({val_acc_threshold_percent:.2f}%)，提前结束训练')
            break
    
    print(f'训练完成。最佳验证准确率: {best_val_acc:.2f}%')
    wandb.finish()
    
    # 如果有最佳模型，加载它
    if best_model_path:
        print(f'加载最佳模型 {best_model_path}')
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return best_val_acc, model

def evaluate_model(model, val_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    return val_loss, val_acc


