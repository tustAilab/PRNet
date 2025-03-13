import torch
from torch.utils.data import DataLoader
import dataset
import model
import train
import config
import os

# 设置环境变量以减少显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def main():
    # 获取完整数据集
    full_dataset = dataset.get_full_dataset()
    
    # 进行5折交叉验证
    all_train_losses, all_val_losses, all_val_accuracies = train.cross_validate(model.ProteinRankNet, full_dataset, n_splits=5, num_epochs=30)
    
    # 输出所有折的结果
    for fold in range(5):
        print(f'Fold {fold + 1} Results:')
        for epoch in range(30):
            print(f'Epoch {epoch + 1} - Train Loss: {all_train_losses[fold][epoch]:.4f}, Val Loss: {all_val_losses[fold][epoch]:.4f}, Val Accuracy: {all_val_accuracies[fold][epoch]:.4f}')

    # 加载最佳模型
    best_model = model.ProteinRankNet(input_shape=(157542,))
    best_model.load_state_dict(torch.load('best_model.pth'))
    print("Best model loaded with validation accuracy:", max([max(fold) for fold in all_val_accuracies]))

if __name__ == "__main__":
    main()