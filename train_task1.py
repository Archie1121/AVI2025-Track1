import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from dataset.baseline_dataset import MultimodalDatasetForTrainT1
from dataset.baseline_dataset import collate_fn_train
from tqdm import tqdm, trange
from model.builder import FusionModel
import json
from datetime import datetime
import pandas as pd


def print_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}\n")

    print("Layer-wise trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<60} {list(param.shape)}  -> {param.numel():,}")


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, desc="Training", leave=False)
    for features, mask, labels in train_bar:
        features = {k: v.to(device) for k, v in features.items()}
        audio_feat = features['audio']
        video_feat = features['video']
        text_feat = features['text']

        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(audio_feat, video_feat, text_feat)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)



def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, predictions, targets = 0, [], []
    val_bar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for features, mask, labels in val_bar:
            features = {k: v.to(device) for k, v in features.items()}
            audio_feat = features['audio']
            video_feat = features['video']
            text_feat = features['text']
            labels = labels.to(device)
            outputs = model(audio_feat, video_feat, text_feat)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            predictions.append(outputs.squeeze().cpu().numpy())
            targets.append(labels.cpu().numpy())
            val_bar.set_postfix(val_loss=loss.item())
    return total_loss / len(loader), mean_squared_error(np.concatenate(targets), np.concatenate(predictions))


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def save_ema_model(model, ema, path):
    ema.apply_shadow(model)
    torch.save(model.state_dict(), path)
    ema.restore(model)
    print(f"EMA model saved to {path}")


def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")



def save_args(args, save_dir="./args_log"):
    os.makedirs(args.args_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args_file = os.path.join(args.args_dir, f"args_{timestamp}.json")
    args_file1 = os.path.join(args.log_dir, f"args_{timestamp}.json")
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)
    with open(args_file1, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Args saved to {args_file},{args_file1}")
    return args_file



def update_result_csv(args, test_loss):
    csv_path = args.csv_dir
    audio_name = os.path.basename(args.audio_dir.rstrip('/'))
    video_name = os.path.basename(args.video_dir.rstrip('/'))
    text_name  = os.path.basename(args.text_dir.rstrip('/'))

    q_map = {'q3': 'q3', 'q4': 'q4', 'q5': 'q5', 'q6': 'q6'}
    q_col = q_map.get(args.question.lower(), None)
    if q_col is None:
        print(f"Unknown question type: {args.question}")
        return

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['audio', 'video', 'text', 'q3', 'q4', 'q5', 'q6'])


    row_idx = df[(df['audio'] == audio_name) &   
                 (df['video'] == video_name) & 
                 (df['text'] == text_name)].index

    if len(row_idx) > 0:
        idx = row_idx[0]
        df.at[idx, q_col] = round(float(test_loss * 16), 4)
    else:
        new_row = {
            'audio': audio_name,
            'video': video_name,
            'text': text_name,
            'q3': '', 'q4': '', 'q5': '', 'q6': ''
        }
        new_row[q_col] = round(float(test_loss * 16), 4)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Results updated in {csv_path}")



def main():
    parser = argparse.ArgumentParser()
    #### for dataset
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--label_col', required=True)

    #### for input_features
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--text_dir', required=True)
    parser.add_argument('--audio_dim', type=int, default=384)
    parser.add_argument('--video_dim', type=int, default=512)
    parser.add_argument('--text_dim', type=int, default=768)

    #### for training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool,default=True)
    parser.add_argument('--optim', type=str, default='adamw')

    #### for model
    # for projector
    parser.add_argument('--HCPdropout_audio', type=float, default=0.2)
    parser.add_argument('--HCPdropout_video', type=float, default=0.2)
    parser.add_argument('--HCPdropout_text', type=float, default=0.2)
    parser.add_argument('--HCPdropout_pure_text', type=float, default=0.1) 
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--unified_dim', type=int, default=512)
    # for AT_VT connector
    parser.add_argument('--heads_num', type=int, default=4)     
    parser.add_argument('--ATCdropout', type=float, default=0.3)
    parser.add_argument('--VTCdropout', type=float, default=0.3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    # for text feature enhancer
    parser.add_argument('--enhancer_dim', type=int, default=512)
    parser.add_argument('--TFEdropout', type=float, default=0.2)
    # for regression head
    parser.add_argument('--RHdropout', type=float, default=0.2)
    parser.add_argument('--target_dim', type=int, default=1)
    parser.add_argument('--num_modalities', type=int, default=3)
    parser.add_argument('--modalities', type=str, default="audio,video,text")
    parser.add_argument('--output_model', default='best_model.pth')
    parser.add_argument('--loss_plot_path', type=str, default='./img/loss_curve.png')
    parser.add_argument('--log_dir', type=str, default='./AVI/logs')
    parser.add_argument('--args_dir', type=str, default='./AVI/logs')
    parser.add_argument('--csv_dir', type=str, default='./AVI/logs')
    parser.add_argument('--training_time', type=str, default='2025-5-23')
    args = parser.parse_args()
    args.modalities = [m.strip() for m in args.modalities.split(',')]
    args_file_path = save_args(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = MultimodalDatasetForTrainT1(args.train_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.label_col, args)
    val_set = MultimodalDatasetForTrainT1(args.val_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.label_col, args)
    test_set = MultimodalDatasetForTrainT1(args.test_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.label_col, args)

    print(f"train_set size: {len(train_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_train, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_train)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_train)

    model = FusionModel(args).to(device)
    criterion = nn.MSELoss()

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5,min_lr=1e-6)


    print("Training started...")
    train_losses, val_losses = [], []
    p_train_losses, p_val_losses = [],[]
    best_val_loss = float('inf')

    for epoch in trange(args.num_epochs, desc="Epochs", ncols=100):
    
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_mse = evaluate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        p_train_loss = train_loss * 16  # for 1-5 scale
        p_val_loss = val_loss * 16
        p_val_mse = val_mse * 16
    
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        p_train_losses.append(p_train_loss)
        p_val_losses.append(p_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, args.output_model)
        

        tqdm.write(f"[Epoch {epoch+1}/{args.num_epochs}] "
               f"Train Loss: {p_train_loss:.4f} | Val Loss: {p_val_loss:.4f} | Val MSE: {p_val_mse:.4f}")


    save_loss_plot(p_train_losses, p_val_losses, args.loss_plot_path)

    # test the model
    model.load_state_dict(torch.load(args.output_model))  # general
    model.eval()
    test_loss, test_mse = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss*16:.4f}, Test MSE: {test_mse*16:.4f}")
    update_result_csv(args,test_loss)

    with open(args_file_path, 'r') as f:
        args_data = json.load(f)
    args_data['test_loss'] = float(test_loss * 16)
    args_data['test_mse'] = float(test_mse * 16)
    with open(args_file_path, 'w') as f:
        json.dump(args_data, f, indent=4)
    print(f"Test results added to {args_file_path}")


if __name__ == '__main__':
    main()
