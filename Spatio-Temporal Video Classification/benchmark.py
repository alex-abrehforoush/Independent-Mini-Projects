"""
UCF50 Video Classification - Complete Benchmark Suite
Trains all 5 architectures and generates comprehensive comparison reports
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SET RANDOM SEEDS
# ============================================================================
mySeed = 42
np.random.seed(mySeed)
random.seed(mySeed)
torch.manual_seed(mySeed)
torch.cuda.manual_seed(mySeed)
torch.backends.cudnn.deterministic = True

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Data parameters
    DATA_PATH = 'UCF50'  # Update this path
    NUM_FRAMES = 20
    IMG_SIZE = (112, 112)
    
    # Class range - Change for each run
    CLASS_START = 0
    CLASS_END = 10
    
    # Training parameters
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Results directory
    RESULTS_DIR = 'benchmark_results'
    
    # Models to benchmark
    MODELS_TO_TRAIN = ['single_frame', 'early_fusion', 'late_fusion', 'cnn_lstm', 'conv_lstm']

config = Config()
os.makedirs(config.RESULTS_DIR, exist_ok=True)

# ============================================================================
# DATASET CLASS
# ============================================================================
class UCF50Dataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, num_frames=20):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        
        for i in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((config.IMG_SIZE[0], config.IMG_SIZE[1], 3), dtype=np.uint8))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, config.IMG_SIZE)
                frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.load_video(video_path)
        
        if self.transform:
            frames = np.stack([self.transform(frame) for frame in frames])
        else:
            frames = torch.FloatTensor(frames).permute(0, 3, 1, 2) / 255.0
        
        return frames, label

# ============================================================================
# DATA LOADING
# ============================================================================
def load_ucf50_data(data_path, class_start=0, class_end=10):
    data_path = Path(data_path)
    video_paths = []
    labels = []
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    
    selected_classes = class_names[class_start:class_end]
    class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
    
    print(f"\n{'='*60}")
    print(f"Loading classes {class_start} to {class_end-1}")
    print(f"Classes: {selected_classes}")
    print(f"{'='*60}\n")
    
    for class_name in selected_classes:
        class_path = data_path / class_name
        for video_file in class_path.glob('*.avi'):
            video_paths.append(str(video_file))
            labels.append(class_to_idx[class_name])
    
    return video_paths, labels, len(selected_classes), selected_classes

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class SingleFrameCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]
        x = x.view(batch_size * num_frames, *x.shape[2:])
        features = self.features(x)
        features = features.view(batch_size, num_frames, -1)
        logits = self.fc(features)
        output = logits.mean(dim=1)
        return output

class EarlyFusionCNN(nn.Module):
    def __init__(self, num_classes, num_frames=20):
        super().__init__()
        self.num_frames = num_frames
        self.conv1 = nn.Conv2d(num_frames * 3, 64, kernel_size=7, stride=2, padding=3)
        resnet = models.resnet18(pretrained=True)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, *x.shape[3:])
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LateFusionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]
        x = x.view(batch_size * num_frames, *x.shape[2:])
        features = self.features(x)
        features = features.view(batch_size, num_frames, -1)
        fused_features = features.mean(dim=1)
        output = self.fc(fused_features)
        return output

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]
        x = x.view(batch_size * num_frames, *x.shape[2:])
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        output = self.fc(h_n[-1])
        return output

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding
        )
        
    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur

class ConvLSTM(nn.Module):
    def __init__(self, num_classes, hidden_channels=64):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.convlstm1 = ConvLSTMCell(32, hidden_channels, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]
        x0 = self.conv1(x[:, 0])
        h, w = x0.shape[2], x0.shape[3]
        
        h1 = torch.zeros(batch_size, self.hidden_channels, h, w).to(x.device)
        c1 = torch.zeros(batch_size, self.hidden_channels, h, w).to(x.device)
        h2 = torch.zeros(batch_size, self.hidden_channels, h, w).to(x.device)
        c2 = torch.zeros(batch_size, self.hidden_channels, h, w).to(x.device)
        
        for t in range(num_frames):
            x_t = self.conv1(x[:, t])
            h1, c1 = self.convlstm1(x_t, (h1, c1))
            h2, c2 = self.convlstm2(h1, (h2, c2))
        
        output = self.avgpool(h2)
        output = output.view(batch_size, -1)
        output = self.fc(output)
        return output

def get_model(model_type, num_classes):
    if model_type == 'single_frame':
        return SingleFrameCNN(num_classes)
    elif model_type == 'early_fusion':
        return EarlyFusionCNN(num_classes, config.NUM_FRAMES)
    elif model_type == 'late_fusion':
        return LateFusionCNN(num_classes)
    elif model_type == 'cnn_lstm':
        return CNNLSTM(num_classes)
    elif model_type == 'conv_lstm':
        return ConvLSTM(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device, return_predictions=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    acc = 100. * correct / total
    loss = running_loss / len(dataloader)
    
    if return_predictions:
        return loss, acc, np.array(all_predictions), np.array(all_labels)
    return loss, acc

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_training_history(history, model_name, save_path):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plots(results_df, save_dir):
    """Create comprehensive comparison plots"""
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    bars = plt.bar(results_df['Model'], results_df['Best Val Acc'], color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Training Time Comparison
    plt.subplot(1, 2, 2)
    bars = plt.bar(results_df['Model'], results_df['Training Time (min)'], color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('Time (minutes)', fontsize=12, fontweight='bold')
    plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Parameters vs Accuracy Scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Parameters (M)'], results_df['Best Val Acc'], 
                s=300, c=results_df['Training Time (min)'], cmap='plasma', 
                edgecolors='black', linewidth=2, alpha=0.7)
    
    for idx, row in results_df.iterrows():
        plt.annotate(row['Model'], (row['Parameters (M)'], row['Best Val Acc']),
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    plt.xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    plt.ylabel('Best Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Complexity vs Performance', fontsize=14, fontweight='bold')
    plt.colorbar(label='Training Time (min)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/params_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Efficiency Score (Accuracy / Training Time)
    plt.figure(figsize=(12, 6))
    efficiency = results_df['Best Val Acc'] / results_df['Training Time (min)']
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(results_df)))
    bars = plt.bar(results_df['Model'], efficiency, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('Efficiency Score (Acc/Time)', fontsize=12, fontweight='bold')
    plt.title('Model Efficiency (Higher is Better)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/efficiency_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Radar Chart
    categories = ['Accuracy', 'Speed', 'Simplicity']
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, row in results_df.iterrows():
        # Normalize values to 0-100 scale
        acc_norm = row['Best Val Acc']
        speed_norm = 100 - (row['Training Time (min)'] / results_df['Training Time (min)'].max() * 100)
        simple_norm = 100 - (row['Parameters (M)'] / results_df['Parameters (M)'].max() * 100)
        
        values = [acc_norm, speed_norm, simple_norm]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('Model Performance Radar', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(results_df, save_dir):
    """Create beautiful summary table"""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for idx, row in results_df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['Parameters (M)']:.2f}M",
            f"{row['Best Val Acc']:.2f}%",
            f"{row['Best Train Acc']:.2f}%",
            f"{row['Final Val Loss']:.4f}",
            f"{row['Training Time (min)']:.1f}m",
            f"{row['Best Epoch']}"
        ])
    
    headers = ['Model', 'Parameters', 'Val Acc', 'Train Acc', 'Val Loss', 'Time', 'Best Epoch']
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colColours=['#4CAF50']*len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color best values
    best_acc_idx = results_df['Best Val Acc'].idxmax() + 1
    best_time_idx = results_df['Training Time (min)'].idxmin() + 1
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best accuracy
    table[(best_acc_idx, 2)].set_facecolor('#FFD700')
    table[(best_acc_idx, 2)].set_text_props(weight='bold')
    
    # Highlight fastest training
    table[(best_time_idx, 5)].set_facecolor('#87CEEB')
    table[(best_time_idx, 5)].set_text_props(weight='bold')
    
    plt.title('Model Benchmark Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{save_dir}/summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# TRAINING PIPELINE
# ============================================================================
def train_model(model_type, train_loader, val_loader, num_classes, class_names, save_dir):
    """Train a single model and return results"""
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*60}\n")
    
    model = get_model(model_type, num_classes).to(config.DEVICE)
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Model Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    # Training loop
    for epoch in range(config.EPOCHS):
        print(f'\nEpoch [{epoch+1}/{config.EPOCHS}]')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, f'{save_dir}/{model_type}_best.pth')
            print(f'✓ Saved best model (Acc: {best_acc:.2f}%)')
    
    training_time = (time.time() - start_time) / 60
    
    # Load best model for final evaluation
    checkpoint = torch.load(f'{save_dir}/{model_type}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get predictions for confusion matrix
    _, final_acc, predictions, true_labels = validate(model, val_loader, criterion, config.DEVICE, return_predictions=True)
    
    # Plot training history
    plot_training_history(history, model_type.replace('_', ' ').title(), 
                         f'{save_dir}/{model_type}_history.png')
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, class_names, 
                         model_type.replace('_', ' ').title(),
                         f'{save_dir}/{model_type}_confusion.png')
    
    print(f"\n✓ {model_type.upper()} completed!")
    print(f"  Best Val Acc: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Training Time: {training_time:.1f} minutes")
    print(f"  Parameters: {num_params/1e6:.2f}M")
    
    return {
        'model_type': model_type,
        'best_val_acc': best_acc,
        'best_train_acc': history['train_acc'][best_epoch-1],
        'final_val_loss': history['val_loss'][-1],
        'best_epoch': best_epoch,
        'training_time': training_time,
        'num_params': num_params / 1e6,
        'history': history
    }

# ============================================================================
# MAIN BENCHMARK FUNCTION
# ============================================================================
def run_benchmark():
    """Run complete benchmark for all models"""
    print(f"\n{'#'*60}")
    print(f"# UCF50 VIDEO CLASSIFICATION BENCHMARK")
    print(f"# Device: {config.DEVICE}")
    print(f"# Models: {', '.join(config.MODELS_TO_TRAIN)}")
    print(f"{'#'*60}\n")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{config.RESULTS_DIR}/run_{config.CLASS_START}_{config.CLASS_END}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Load data
    video_paths, labels, num_classes, class_names = load_ucf50_data(
        config.DATA_PATH, config.CLASS_START, config.CLASS_END
    )
    
    # Train-test split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=mySeed, stratify=labels
    )
    
    print(f"Dataset Statistics:")
    print(f"  Total videos: {len(video_paths)}")
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    print(f"  Number of classes: {num_classes}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = UCF50Dataset(train_paths, train_labels, train_transform, config.NUM_FRAMES)
    val_dataset = UCF50Dataset(val_paths, val_labels, val_transform, config.NUM_FRAMES)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    # Train all models
    all_results = []
    
    for model_type in config.MODELS_TO_TRAIN:
        try:
            result = train_model(model_type, train_loader, val_loader, 
                               num_classes, class_names, run_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error training {model_type}: {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Model': [r['model_type'].replace('_', ' ').title() for r in all_results],
        'Best Val Acc': [r['best_val_acc'] for r in all_results],
        'Best Train Acc': [r['best_train_acc'] for r in all_results],
        'Final Val Loss': [r['final_val_loss'] for r in all_results],
        'Best Epoch': [r['best_epoch'] for r in all_results],
        'Training Time (min)': [r['training_time'] for r in all_results],
        'Parameters (M)': [r['num_params'] for r in all_results]
    })
    
    # Sort by accuracy
    results_df = results_df.sort_values('Best Val Acc', ascending=False).reset_index(drop=True)
    
    # Save results
    results_df.to_csv(f'{run_dir}/benchmark_results.csv', index=False)
    
    # Save detailed results as JSON
    with open(f'{run_dir}/detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=4, default=str)
    
    # Create visualizations
    print(f"\n{'='*60}")
    print("Creating Comparison Visualizations...")
    print(f"{'='*60}\n")
    
    create_comparison_plots(results_df, run_dir)
    create_summary_table(results_df, run_dir)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*60}\n")
    print(results_df.to_string(index=False))
    
    print(f"\n📊 Results saved to: {run_dir}")
    print(f"\n🏆 Best Model: {results_df.iloc[0]['Model']} ({results_df.iloc[0]['Best Val Acc']:.2f}%)")
    print(f"⚡ Fastest Model: {results_df.loc[results_df['Training Time (min)'].idxmin(), 'Model']}")
    print(f"💾 Most Efficient: {results_df.loc[results_df['Parameters (M)'].idxmin(), 'Model']}")
    
    # Generate HTML report
    generate_html_report(results_df, all_results, run_dir, class_names)
    
    return results_df, all_results

def generate_html_report(results_df, all_results, run_dir, class_names):
    """Generate comprehensive HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UCF50 Benchmark Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            h1 {{
                color: #667eea;
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 2px;
            }}
            .subtitle {{
                text-align: center;
                color: #666;
                font-size: 1.2em;
                margin-bottom: 40px;
            }}
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                transition: transform 0.3s;
            }}
            .summary-card:hover {{
                transform: translateY(-5px);
            }}
            .summary-card h3 {{
                margin: 0 0 10px 0;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .summary-card .value {{
                font-size: 2.5em;
                font-weight: bold;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 30px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-size: 0.9em;
            }}
            td {{
                padding: 15px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .best {{
                background-color: #ffd700 !important;
                font-weight: bold;
            }}
            .images-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 30px;
                margin: 30px 0;
            }}
            .image-container {{
                border: 2px solid #ddd;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .image-container img {{
                width: 100%;
                display: block;
            }}
            .image-title {{
                background: #667eea;
                color: white;
                padding: 15px;
                text-align: center;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .section {{
                margin: 50px 0;
            }}
            .section-title {{
                font-size: 2em;
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            .info-box {{
                background: #f8f9fa;
                border-left: 5px solid #667eea;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 2px solid #ddd;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 UCF50 Video Classification Benchmark</h1>
            <div class="subtitle">
                Classes {config.CLASS_START}-{config.CLASS_END-1} | {len(all_results)} Models Trained | {config.NUM_FRAMES} Frames
            </div>
            
            <div class="summary">
                <div class="summary-card">
                    <h3>🏆 Best Model</h3>
                    <div class="value">{results_df.iloc[0]['Model']}</div>
                    <div>{results_df.iloc[0]['Best Val Acc']:.2f}% Accuracy</div>
                </div>
                <div class="summary-card">
                    <h3>⚡ Fastest Training</h3>
                    <div class="value">{results_df['Training Time (min)'].min():.1f}m</div>
                    <div>{results_df.loc[results_df['Training Time (min)'].idxmin(), 'Model']}</div>
                </div>
                <div class="summary-card">
                    <h3>💾 Smallest Model</h3>
                    <div class="value">{results_df['Parameters (M)'].min():.1f}M</div>
                    <div>{results_df.loc[results_df['Parameters (M)'].idxmin(), 'Model']}</div>
                </div>
                <div class="summary-card">
                    <h3>📊 Avg Accuracy</h3>
                    <div class="value">{results_df['Best Val Acc'].mean():.1f}%</div>
                    <div>Across all models</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">📈 Results Table</div>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Val Accuracy</th>
                        <th>Train Accuracy</th>
                        <th>Parameters</th>
                        <th>Training Time</th>
                        <th>Best Epoch</th>
                    </tr>
    """
    
    best_acc_idx = results_df['Best Val Acc'].idxmax()
    for idx, row in results_df.iterrows():
        row_class = 'best' if idx == best_acc_idx else ''
        html_content += f"""
                    <tr class="{row_class}">
                        <td>{idx + 1}</td>
                        <td><strong>{row['Model']}</strong></td>
                        <td>{row['Best Val Acc']:.2f}%</td>
                        <td>{row['Best Train Acc']:.2f}%</td>
                        <td>{row['Parameters (M)']:.2f}M</td>
                        <td>{row['Training Time (min)']:.1f} min</td>
                        <td>{row['Best Epoch']}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="info-box">
                <strong>📝 Dataset Information:</strong><br>
                Classes: """ + ", ".join(class_names) + """<br>
                Frames per video: """ + str(config.NUM_FRAMES) + """<br>
                Random Seed: """ + str(mySeed) + """<br>
                Device: """ + str(config.DEVICE) + """
            </div>
            
            <div class="section">
                <div class="section-title">📊 Comparison Visualizations</div>
                <div class="images-grid">
                    <div class="image-container">
                        <div class="image-title">Accuracy & Time Comparison</div>
                        <img src="comparison_bars.png" alt="Comparison">
                    </div>
                    <div class="image-container">
                        <div class="image-title">Model Efficiency</div>
                        <img src="efficiency_score.png" alt="Efficiency">
                    </div>
                    <div class="image-container">
                        <div class="image-title">Performance vs Complexity</div>
                        <img src="params_vs_accuracy.png" alt="Complexity">
                    </div>
                    <div class="image-container">
                        <div class="image-title">Performance Radar</div>
                        <img src="radar_chart.png" alt="Radar">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">📉 Training Curves</div>
                <div class="images-grid">
    """
    
    for result in all_results:
        model_type = result['model_type']
        html_content += f"""
                    <div class="image-container">
                        <div class="image-title">{model_type.replace('_', ' ').title()}</div>
                        <img src="{model_type}_history.png" alt="{model_type} history">
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">🔥 Confusion Matrices</div>
                <div class="images-grid">
    """
    
    for result in all_results:
        model_type = result['model_type']
        html_content += f"""
                    <div class="image-container">
                        <div class="image-title">{model_type.replace('_', ' ').title()}</div>
                        <img src="{model_type}_confusion.png" alt="{model_type} confusion">
                    </div>
        """
    
    html_content += f"""
                </div>
            </div>
            
            <div class="footer">
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>UCF50 Video Classification Benchmark | PyTorch Implementation</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(f'{run_dir}/benchmark_report.html', 'w') as f:
        f.write(html_content)
    
    print(f"\n📄 HTML Report generated: {run_dir}/benchmark_report.html")

# ============================================================================
# RUN ALL 5 CLASS RANGES
# ============================================================================
def run_all_class_ranges():
    """Run benchmark for all 5 class ranges"""
    class_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    all_runs_results = []
    
    print("\n" + "="*80)
    print("RUNNING COMPLETE BENCHMARK FOR ALL 5 CLASS RANGES")
    print("="*80 + "\n")
    
    for start, end in class_ranges:
        config.CLASS_START = start
        config.CLASS_END = end
        
        print(f"\n{'#'*80}")
        print(f"# RUN {len(all_runs_results) + 1}/5: Classes {start}-{end-1}")
        print(f"{'#'*80}\n")
        
        results_df, all_results = run_benchmark()
        all_runs_results.append({
            'range': f'{start}-{end-1}',
            'results': results_df,
            'details': all_results
        })
    
    # Create aggregate report
    print("\n" + "="*80)
    print("CREATING AGGREGATE REPORT")
    print("="*80 + "\n")
    
    aggregate_dir = f"{config.RESULTS_DIR}/aggregate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(aggregate_dir, exist_ok=True)
    
    # Aggregate results across all runs
    aggregate_results = {}
    for run_data in all_runs_results:
        for _, row in run_data['results'].iterrows():
            model = row['Model']
            if model not in aggregate_results:
                aggregate_results[model] = {
                    'accuracies': [],
                    'times': [],
                    'params': row['Parameters (M)']
                }
            aggregate_results[model]['accuracies'].append(row['Best Val Acc'])
            aggregate_results[model]['times'].append(row['Training Time (min)'])
    
    # Create aggregate DataFrame
    aggregate_df = pd.DataFrame({
        'Model': list(aggregate_results.keys()),
        'Mean Accuracy': [np.mean(v['accuracies']) for v in aggregate_results.values()],
        'Std Accuracy': [np.std(v['accuracies']) for v in aggregate_results.values()],
        'Min Accuracy': [np.min(v['accuracies']) for v in aggregate_results.values()],
        'Max Accuracy': [np.max(v['accuracies']) for v in aggregate_results.values()],
        'Mean Time (min)': [np.mean(v['times']) for v in aggregate_results.values()],
        'Parameters (M)': [v['params'] for v in aggregate_results.values()]
    }).sort_values('Mean Accuracy', ascending=False)
    
    # Save aggregate results
    aggregate_df.to_csv(f'{aggregate_dir}/aggregate_results.csv', index=False)
    
    # Plot aggregate comparison
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Mean accuracy with error bars
    ax = axes[0, 0]
    x_pos = np.arange(len(aggregate_df))
    ax.bar(x_pos, aggregate_df['Mean Accuracy'], yerr=aggregate_df['Std Accuracy'],
           capsize=5, color=plt.cm.viridis(np.linspace(0, 1, len(aggregate_df))),
           edgecolor='black', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(aggregate_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Accuracy Across All Runs', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Accuracy range
    ax = axes[0, 1]
    for idx, row in aggregate_df.iterrows():
        ax.plot([idx, idx], [row['Min Accuracy'], row['Max Accuracy']], 
               'o-', linewidth=3, markersize=8, label=row['Model'])
    ax.set_xticks(range(len(aggregate_df)))
    ax.set_xticklabels(aggregate_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Range (Min-Max)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Training time comparison
    ax = axes[1, 0]
    ax.bar(range(len(aggregate_df)), aggregate_df['Mean Time (min)'],
           color=plt.cm.plasma(np.linspace(0, 1, len(aggregate_df))),
           edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(aggregate_df)))
    ax.set_xticklabels(aggregate_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Training Time', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Accuracy across runs
    ax = axes[1, 1]
    for model in aggregate_results.keys():
        accuracies = aggregate_results[model]['accuracies']
        ax.plot(range(1, 6), accuracies, 'o-', linewidth=2, markersize=8, label=model)
    ax.set_xlabel('Run Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Consistency Across Runs', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))
    
    plt.tight_layout()
    plt.savefig(f'{aggregate_dir}/aggregate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print aggregate summary
    print("\n" + "="*80)
    print("AGGREGATE RESULTS SUMMARY")
    print("="*80 + "\n")
    print(aggregate_df.to_string(index=False))
    
    print(f"\n📊 Aggregate results saved to: {aggregate_dir}")
    print(f"\n🏆 Overall Best Model: {aggregate_df.iloc[0]['Model']}")
    print(f"   Mean Accuracy: {aggregate_df.iloc[0]['Mean Accuracy']:.2f}% ± {aggregate_df.iloc[0]['Std Accuracy']:.2f}%")
    
    return aggregate_df, all_runs_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='UCF50 Video Classification Benchmark')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'all'],
                       help='Run single class range or all 5 ranges')
    parser.add_argument('--class_start', type=int, default=0,
                       help='Starting class index (for single mode)')
    parser.add_argument('--class_end', type=int, default=10,
                       help='Ending class index (for single mode)')
    parser.add_argument('--models', nargs='+', 
                       default=['single_frame', 'early_fusion', 'late_fusion', 'cnn_lstm', 'conv_lstm'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    config.CLASS_START = args.class_start
    config.CLASS_END = args.class_end
    config.MODELS_TO_TRAIN = args.models
    
    if args.mode == 'single':
        # Run single class range
        results_df, all_results = run_benchmark()
    else:
        # Run all 5 class ranges
        aggregate_df, all_runs = run_all_class_ranges()