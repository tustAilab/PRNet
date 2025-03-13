import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import config
from encoding_tools import *  # 确保此处的路径正确，导入新的编码函数
import pickle

# 自定义数据集类，用于存储蛋白质序列和标签
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# 从CSV文件加载数据
def load_data(filepath):
    data = pd.read_csv(filepath)
    sequences = data['Aligned_amino_acid_sequence'].values
    labels = data['red_norm'].values
    return sequences, labels

# 新的独热编码过程
def encode_sequence(sequence, ss, contacts):
    return one_hot_([sequence], ss, contacts)[0]  # 使用新的one_hot_编码函数进行编码

# 预处理数据，包括编码和标准化
def prepare_data(sequences, labels, ss, contacts):
    # 对每个序列进行编码
    encoded_sequences = [encode_sequence(seq, ss, contacts) for seq in sequences]
    
    encoded_sequences = np.array(encoded_sequences, dtype=float)
    labels = np.array(labels, dtype=float)
    
    # 标准化数据
    mean = np.mean(encoded_sequences, axis=0)
    std = np.std(encoded_sequences, axis=0)
    encoded_sequences = (encoded_sequences - mean) / (std + 1e-8)
    
    return encoded_sequences, labels

# 获取数据集并进行五折交叉验证的划分
def get_datasets(fold=0, num_folds=5):
    sequences, labels = load_data(config.DATASET_PATH)
    
    # 加载空间关系文件
    with open(config.CONTACT_MAP_PATH, 'rb') as f:
        ss, contacts = pickle.load(f)
    
    # 编码和标准化处理
    encoded_sequences, labels = prepare_data(sequences, labels, ss, contacts)
    print(len(contacts)) 
    # 计算每个折的起始和结束索引
    fold_size = len(encoded_sequences) // num_folds
    start_idx = fold * fold_size
    if fold == num_folds - 1:
        end_idx = len(encoded_sequences)
    else:
        end_idx = start_idx + fold_size

    # 划分训练集和验证集
    train_sequences = np.concatenate([
        encoded_sequences[:start_idx],  # 当前折之前的所有数据
        encoded_sequences[end_idx:]     # 当前折之后的所有数据
    ])
    train_labels = np.concatenate([
        labels[:start_idx],  # 当前折之前的所有标签
        labels[end_idx:]     # 当前折之后的所有标签
    ])
    val_sequences = encoded_sequences[start_idx:end_idx]  # 当前折的数据
    val_labels = labels[start_idx:end_idx]                # 当前折的标签

    # 创建训练集和验证集的数据集对象
    train_dataset = ProteinDataset(train_sequences, train_labels)
    val_dataset = ProteinDataset(val_sequences, val_labels)
    
    return train_dataset, val_dataset

# 获取完整数据集
def get_full_dataset():
    sequences, labels = load_data(config.DATASET_PATH)
    
    # 加载空间关系文件,形成蛋白质序列的接触图
    with open(config.CONTACT_MAP_PATH, 'rb') as f:
        ss, contacts = pickle.load(f)
    
     
    # 编码和标准化处理
    encoded_sequences, labels = prepare_data(sequences, labels, ss, contacts)
    
    full_dataset = ProteinDataset(encoded_sequences, labels)
    
    return full_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = get_datasets(fold=0, num_folds=5)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
      
    # 打印一个训练样本和一个验证样本的形状
    train_sample = train_dataset[0][0]
    val_sample = val_dataset[0][0]
    print(f"Shape of a training sample: {train_sample.shape}")
    print(f"Shape of a validation sample: {val_sample.shape}")
