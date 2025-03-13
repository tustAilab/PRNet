import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import dataset
import model
import pandas as pd
import config
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch.onnx

class PairwiseRankingLoss(nn.Module):  #使用hinge loss
    def __init__(self, margin=0.1):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        # 计算两个输出之间的差异
        diff = output1 - output2
        
        # 根据目标标签调整hinge loss
        #hinge_loss = torch.mean(torch.max(torch.zeros_like(diff), self.margin - target * diff))

        hinge_loss = torch.mean((1 - target) * torch.clamp(self.margin + diff, min=0) + target * torch.clamp(self.margin - diff, min=0))
        return hinge_loss


        return total_loss
def remove_hooks(model):
    for module in model.modules():
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()

def cross_validate(model_class, dataset, n_splits=5, num_epochs=30):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    results = []

    best_val_acc = 0.0
    best_model_wts = None

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='runs/exp1')

    for fold, (train_index, val_index) in enumerate(kf.split(dataset.sequences)):
        train_sequences, val_sequences = dataset.sequences[train_index], dataset.sequences[val_index]
        train_labels, val_labels = dataset.labels[train_index], dataset.labels[val_index]
        
        train_pairs, train_labels = create_pairs(train_sequences, train_labels)
        val_pairs, val_labels = create_pairs(val_sequences, val_labels)
        
        train_loader = DataLoader(list(zip(train_pairs, train_labels)), batch_size=32, shuffle=True)
        val_loader = DataLoader(list(zip(val_pairs, val_labels)), batch_size=32, shuffle=False)

        # Debug information
        print(f"Train loader size: {len(train_loader)}, Validation loader size: {len(val_loader)}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(input_shape=(157542,)).to(device)  # 根据新的编码调整输入形状
        # remove_hooks(model)  # 移除所有 hooks
        criterion = PairwiseRankingLoss(margin=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
        
        scaler = GradScaler()

        # if fold == 0:  # 仅在第一折训练时添加模型结构到 TensorBoard
        #     sample_input = torch.zeros((1, 1, 1, 600747)).to(device)  # 根据新的编码调整输入形状
        #     model.eval()  # 确保模型在评估模式下
        #     torch.onnx.export(model, sample_input, "model.onnx", opset_version=11)
        #     writer.add_graph(model, sample_input)
        
        fold_train_losses = []
        fold_val_losses = []
        fold_val_accuracies = []

        print(f'Fold {fold + 1} - GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB')
        print(f'Fold {fold + 1} - GPU Memory Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB')

        best_epoch_val_acc = 0.0  # 每折最佳验证准确率

        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, scaler, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            fold_val_accuracies.append(val_acc)

            # Log the training and validation loss and accuracy
            writer.add_scalar(f'Fold_{fold+1}/Train_Loss', train_loss, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Val_Loss', val_loss, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Val_Accuracy', val_acc, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = model.state_dict()

            # if val_acc > best_epoch_val_acc:  # 更新每折最佳验证准确率
            #     best_epoch_val_acc = val_acc    

            print(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

            results.append({
                'Fold': fold + 1,
                'Epoch': epoch + 1,
                'Train Loss': train_loss,
                'Val Loss': val_loss,
                'Val Accuracy': val_acc
            })

        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)
        all_val_accuracies.append(fold_val_accuracies)
      
        print(f'Fold {fold + 1} Average Train Loss: {sum(fold_train_losses) / num_epochs:.4f}, Average Val Loss: {sum(fold_val_losses) / num_epochs:.4f}, Average Val Accuracy: {sum(fold_val_accuracies) / num_epochs:.4f}')

        # 手动清理显存
        del model, train_loader, val_loader, optimizer, scaler
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    results_df.to_csv('cross_validation_results.csv', index=False)

    if best_model_wts is not None:
        torch.save(best_model_wts, 'best_model.pth')

    writer.close()

    return all_train_losses, all_val_losses, all_val_accuracies

def train(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for (seq1, seq2), labels in tqdm(train_loader, desc="Training", leave=False):
        seq1, seq2, labels = seq1.to(device).float(), seq2.to(device).float(), labels.to(device).float()
        
        optimizer.zero_grad()
        with autocast():
            outputs1 = model(seq1)
            outputs2 = model(seq2)
            loss = criterion(outputs1, outputs2, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * seq1.size(0)
        
        # Debug information
        # print(f"Outputs1 shape: {outputs1.shape}, Outputs2 shape: {outputs2.shape}, Labels shape: {labels.shape}")
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for (seq1, seq2), labels in tqdm(val_loader, desc="Validation", leave=False):
            seq1, seq2, labels = seq1.to(device).float(), seq2.to(device).float(), labels.to(device).float()
            
            with autocast():
                outputs1 = model(seq1)
                outputs2 = model(seq2)
                loss = criterion(outputs1, outputs2, labels)
            
            running_loss += loss.item() * seq1.size(0)
            batch_outputs = (outputs1 > outputs2).cpu().numpy().flatten().tolist()
            if len(batch_outputs) != len(labels):
                print(f"Mismatch in batch sizes: {len(batch_outputs)} outputs vs {len(labels)} labels")
            all_outputs.extend(batch_outputs)
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

            # # 调试信息
            # print(f"Batch outputs length: {len(batch_outputs)}, Labels length: {len(labels.cpu().numpy().flatten().tolist())}")
            # print(f"All outputs so far length: {len(all_outputs)}, All labels so far length: {len(all_labels)}")

            
    epoch_loss = running_loss / len(val_loader.dataset)
    
    # Ensure lengths are the same before calculating accuracy
    # min_length = min(len(all_outputs), len(all_labels))
    # all_outputs = all_outputs[:min_length]
    # all_labels = all_labels[:min_length]
    
    # print(f"All outputs length: {len(all_outputs)}, All labels length: {len(all_labels)}")

    epoch_acc = accuracy_score(all_labels, all_outputs)
    return epoch_loss, epoch_acc


def create_pairs(data, labels):
    pairs = []
    pair_labels = []
    num_samples = len(data)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            pairs.append((data[i], data[j]))
            if labels[i] > labels[j]:
                pair_labels.append(1)
            else:
                pair_labels.append(0)

    return pairs, pair_labels

if __name__ == "__main__":
    full_dataset = dataset.get_full_dataset()
    all_train_losses, all_val_losses, all_val_accuracies = cross_validate(model.ProteinRankNet, full_dataset, n_splits=5, num_epochs=30)