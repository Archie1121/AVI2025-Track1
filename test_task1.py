import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.builder import FusionModel
from dataset.baseline_dataset import MultimodalDatasetForTestT1, collate_fn_test


def predict(model, dataloader, device):
    model.eval()
    preds, ids = [], []
    with torch.no_grad():
        for features, _, sample_ids in tqdm(dataloader, desc="🔍 Predicting"):
            features = {k: v.to(device) for k, v in features.items()}
            outputs = model(features['audio'], features['video'], features['text'])  # [B, 1]
            outputs = outputs.squeeze().cpu().numpy()
            outputs = outputs * 4 + 1  
            preds.extend(outputs.tolist())
            ids.extend(sample_ids)
    return ids, preds


def update_submission_csv(csv_path, ids, preds, trait):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['id', 'Honesty-Humility', 'Extraversion', 'Agreeableness', 'Conscientiousness'])

    id2pred = dict(zip(ids, preds))

    for idx, row in df.iterrows():
        if row['id'] in id2pred:
            df.at[idx, trait] = round(id2pred[row['id']], 4)

    known_ids = set(df['id'])
    new_rows = []
    for sid, val in id2pred.items():
        if sid not in known_ids:
            new_rows.append({'id': sid, trait: round(val, 4)})
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"submission has been updated: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_json', required=True, help="args.json")
    parser.add_argument('--submission_csv', required=True, help="submission.csv")
    parser.add_argument('--trait', required=True, choices=['Honesty-Humility', 'Extraversion', 'Agreeableness', 'Conscientiousness'])
    parser.add_argument('--override_test_csv', default=None, help="optional, override the test CSV path")

    args = parser.parse_args()

    # load args from JSON
    with open(args.args_json, 'r') as f:
        config = json.load(f)
    train_args = argparse.Namespace(**config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if args.override_test_csv is provided, use it; otherwise, use the one from train_args
    test_csv_path = args.override_test_csv if args.override_test_csv else train_args.test_csv

    # load the model
    model = FusionModel(train_args).to(device)
    model.load_state_dict(torch.load(train_args.output_model, map_location=device))
    print(f"load model from: {train_args.output_model}")

    test_dataset = MultimodalDatasetForTestT1(
        csv_file=test_csv_path,
        audio_dir=train_args.audio_dir,
        video_dir=train_args.video_dir,
        text_dir=train_args.text_dir,
        question=train_args.question,
        args=train_args
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
        num_workers=train_args.num_workers,
        pin_memory=train_args.pin_memory
    )

    # inference
    ids, preds = predict(model, test_loader, device)

    # update submission CSV
    update_submission_csv(args.submission_csv, ids, preds, args.trait)


if __name__ == '__main__':
    main()
