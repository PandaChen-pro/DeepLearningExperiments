import re
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def extract_metrics_from_log(log_content):
    """从日志内容中提取训练损失、验证损失、训练准确率和验证准确率"""
    epoch_train_pattern = r"Epoch \[(\d+)/\d+\] - Loss: (\d+\.\d+), Accuracy: (\d+\.\d+)%"
    epoch_val_pattern = r"Epoch (\d+)/\d+, Val Loss: (\d+\.\d+), Val Acc: (\d+\.\d+)%"
    
    train_epochs = []
    train_losses = []
    train_accs = []
    for match in re.finditer(epoch_train_pattern, log_content):
        epoch = int(match.group(1))
        loss = float(match.group(2))
        acc = float(match.group(3))
        train_epochs.append(epoch)
        train_losses.append(loss)
        train_accs.append(acc)
    
    val_epochs = []
    val_losses = []
    val_accs = []
    for match in re.finditer(epoch_val_pattern, log_content):
        epoch = int(match.group(1))
        loss = float(match.group(2))
        acc = float(match.group(3))
        val_epochs.append(epoch)
        val_losses.append(loss)
        val_accs.append(acc)
    
    return train_epochs, train_losses, train_accs, val_epochs, val_losses, val_accs

def merge_training_sessions(log_files):
    """合并多个训练会话的日志数据"""
    all_train_epochs = []
    all_train_losses = []
    all_train_accs = []
    all_val_epochs = []
    all_val_losses = []
    all_val_accs = []
    
    epoch_offset = 0  # 用于调整后续训练会话的epoch编号
    
    for i, log_file in enumerate(log_files):
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
        except FileNotFoundError:
            print(f"警告: 找不到日志文件 {log_file}")
            continue
        
        print(f"处理日志文件: {log_file}")
        
        # 提取当前日志的指标数据
        train_epochs, train_losses, train_accs, val_epochs, val_losses, val_accs = extract_metrics_from_log(log_content)
        
        if i > 0:  
            if all_train_epochs:
                epoch_offset = max(all_train_epochs)
            elif all_val_epochs:
                epoch_offset = max(all_val_epochs)
            
            train_epochs = [e + epoch_offset for e in train_epochs]
            val_epochs = [e + epoch_offset for e in val_epochs]
        
        all_train_epochs.extend(train_epochs)
        all_train_losses.extend(train_losses)
        all_train_accs.extend(train_accs)
        all_val_epochs.extend(val_epochs)
        all_val_losses.extend(val_losses)
        all_val_accs.extend(val_accs)
        
        print(f"  训练数据点: {len(train_epochs)}")
        print(f"  验证数据点: {len(val_epochs)}")
    
    return all_train_epochs, all_train_losses, all_train_accs, all_val_epochs, all_val_losses, all_val_accs

def plot_loss_curves(train_epochs, train_losses, val_epochs, val_losses):
    """绘制训练和验证损失曲线并保存"""
    plt.figure(figsize=(12, 7))
    
    plt.plot(train_epochs, train_losses, 'b-', label='Training Loss', marker='o', markersize=4)
    
    plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', marker='s', markersize=4)
    
    plt.title('Training and Validation Loss Over Epochs (Combined Sessions)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    if len(train_epochs) > 1:
        for i in range(1, len(train_epochs)):
            if train_epochs[i] <= train_epochs[i-1]:  
                plt.axvline(x=train_epochs[i-1], color='g', linestyle='--', 
                           label='Session Boundary')
                plt.text(train_epochs[i-1], max(train_losses)*0.98, 'Session Change', 
                        rotation=90, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('loss_curves_combined.png', dpi=300, bbox_inches='tight')
    print("损失曲线已保存为 'loss_curves_combined.png'")
    
    plt.close()

def plot_accuracy_curves(train_epochs, train_accs, val_epochs, val_accs):
    """绘制训练和验证准确率曲线并保存"""
    plt.figure(figsize=(12, 7))
    
    plt.plot(train_epochs, train_accs, 'b-', label='Training Accuracy', marker='o', markersize=4)
    
    plt.plot(val_epochs, val_accs, 'r-', label='Validation Accuracy', marker='s', markersize=4)
    
    plt.title('Training and Validation Accuracy Over Epochs (Combined Sessions)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    if len(train_epochs) > 1:
        for i in range(1, len(train_epochs)):
            if train_epochs[i] <= train_epochs[i-1]:  
                plt.axvline(x=train_epochs[i-1], color='g', linestyle='--', 
                           label='Session Boundary')
                plt.text(train_epochs[i-1], max(train_accs)*0.98, 'Session Change', 
                        rotation=90, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('accuracy_curves_combined.png', dpi=300, bbox_inches='tight')
    print("准确率曲线已保存为 'accuracy_curves_combined.png'")
    
    plt.close()

def plot_individual_metrics(train_epochs, train_losses, train_accs, val_epochs, val_losses, val_accs):
    """分别绘制各指标的曲线"""
    metrics = [
        ('Train Loss', train_epochs, train_losses, 'train_loss_curve.png', 'b'),
        ('Validation Loss', val_epochs, val_losses, 'val_loss_curve.png', 'r'),
        ('Train Accuracy', train_epochs, train_accs, 'train_acc_curve.png', 'g'),
        ('Validation Accuracy', val_epochs, val_accs, 'val_acc_curve.png', 'm')
    ]
    
    for title, epochs, values, filename, color in metrics:
        if not epochs or not values:
            print(f"跳过 {title} 曲线，因为没有数据")
            continue
            
        plt.figure(figsize=(12, 7))
        
        plt.plot(epochs, values, f'{color}-', label=title, marker='o', markersize=4)
        
        y_label = 'Accuracy (%)' if 'Accuracy' in title else 'Loss'
        plt.title(f'{title} Over Epochs (Combined Sessions)', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        if len(epochs) > 1:
            for i in range(1, len(epochs)):
                if epochs[i] <= epochs[i-1]:  
                    plt.axvline(x=epochs[i-1], color='gray', linestyle='--', 
                               label='Session Boundary')
                    plt.text(epochs[i-1], max(values)*0.98, 'Session Change', 
                            rotation=90, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"{title} 曲线已保存为 '{filename}'")
        
        plt.close()

def ensure_output_dir(dir_name="output"):
    """确保输出目录存在"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def main():
    log_files = [
        '/home/code/deep_learning_experiments/utils/data/epoch-300.log', 
        '/home/code/deep_learning_experiments/utils/data/output copy.log',
        '/home/code/deep_learning_experiments/utils/data/after_100.log'  
    ]
    
    if not all(os.path.exists(f) for f in log_files):
        print("警告: 找不到指定的日志文件。尝试查找当前目录下的日志文件...")
        available_logs = glob.glob("*log*.txt")
        log_files = available_logs[:2]  
        print(f"找到日志文件: {log_files}")
       
   
    train_epochs, train_losses, train_accs, val_epochs, val_losses, val_accs = merge_training_sessions(log_files)
    
    print("\n合并后的训练指标数据:")
    for epoch, loss, acc in zip(train_epochs, train_losses, train_accs):
        print(f"Epoch {epoch}: Loss={loss}, Acc={acc}%")
    
    print("\n合并后的验证指标数据:")
    for epoch, loss, acc in zip(val_epochs, val_losses, val_accs):
        print(f"Epoch {epoch}: Loss={loss}, Acc={acc}%")
    
    output_dir = ensure_output_dir()
    
    os.chdir(output_dir)
    
    if train_losses and val_losses:
        plot_loss_curves(train_epochs, train_losses, val_epochs, val_losses)
    else:
        print("未能从日志中提取到足够的损失数据")
        
    if train_accs and val_accs:
        plot_accuracy_curves(train_epochs, train_accs, val_epochs, val_accs)
    else:
        print("未能从日志中提取到足够的准确率数据")
    
    plot_individual_metrics(train_epochs, train_losses, train_accs, val_epochs, val_losses, val_accs)
    
    print(f"所有图表已保存到 '{os.path.abspath(output_dir)}' 目录")

if __name__ == "__main__":
    main()
