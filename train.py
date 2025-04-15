import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, f1_score
from tqdm import tqdm
import pandas as pd
from dataloader import get_dataloaders
from model import CNNClassifier
from sklearn.model_selection import train_test_split
import torch.cuda as cuda

def save_stats(filename, epoch, loss, accuracy, precision, f1):
    with open(filename, 'a') as f:
        f.write(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}, "
                f"Precision={precision:.4f}, F1={f1:.4f}\n")

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data'
    model_dir = 'models'
    stats_dir = 'stats'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(data_dir, 'ugransome.csv'))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

    train_loader, val_loader, _ = get_dataloaders(os.path.join(data_dir, 'train.csv'), batch_size=4096)

    model = CNNClassifier(num_classes=9).to(device)  # Updated to 9 classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2000
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_bar.set_postfix({'loss': running_loss / (len(train_preds) / 32)})

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * sum([p == l for p, l in zip(train_preds, train_labels)]) / len(train_labels)
        train_prec = precision_score(train_labels, train_preds, average='macro', zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)

        save_stats(os.path.join(stats_dir, 'train_stats.txt'), epoch + 1, 
                   train_loss, train_acc, train_prec, train_f1)

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_bar.set_postfix({'loss': val_loss / (len(val_preds) / 32)})

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * sum([p == l for p, l in zip(val_preds, val_labels)]) / len(val_labels)
        val_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        save_stats(os.path.join(stats_dir, 'inference_stats.txt'), epoch + 1, 
                   val_loss, val_acc, val_prec, val_f1)

        print(f"Epoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            print(f"Saved best model with F1={best_f1:.4f}")

if __name__ == "__main__":
    train()