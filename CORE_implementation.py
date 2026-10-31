"""
CORE: Correlation-Guided Representation for Multi-Label Text Classification
A PyTorch/Hugging Face reimplementation scaffold based on:
Zhang et al., IJCAI 2021, "Correlation-Guided Representation for Multi-Label Text Classification".

Accepted data formats:
1) Original uploaded AAPD format: one CSV per split with an `abstract` text
   column and 54 binary label columns (one column per topic).
2) Optional converted format: `text` plus pipe-separated `labels`.

For the original AAPD format, label-column names are used as the label vocabulary,
and the binary values are converted directly into multi-hot targets. The label
columns are mapped to BERT [unusedN] tokens because the paper says to use unused
tokens when label descriptions are unavailable.

Install:
    pip install torch transformers pandas scikit-learn

Example:
    python core_reimplementation.py \
      --train train.csv --valid valid.csv --test test.csv \
      --text-col text --labels-col labels --label-sep "|" \
      --output-dir runs/core --epochs 5

Notes:
- This implements the main CORE mechanism, not every baseline in the paper.
- The paper reports TensorFlow, BERT-base-uncased, batch size 32,
  learning rate 5e-5, window size 10, max total length 320.
- Exact preprocessing/splits should follow the dataset's official protocol.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, hamming_loss,
    roc_auc_score, label_ranking_loss, coverage_error
)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_labels(value, separator="|"):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).split(separator) if x.strip()]


def detect_label_columns(dataframe: pd.DataFrame, text_col: str, labels_col: str):
    """Return label columns for native one-hot/multi-hot AAPD CSVs, or None."""
    if labels_col in dataframe.columns:
        return None
    return [c for c in dataframe.columns if c != text_col]


def labels_from_row(row, label_columns):
    """Convert a row of binary label columns into its set of active labels."""
    active = []
    for col in label_columns:
        value = pd.to_numeric(row[col], errors="coerce")
        if pd.notna(value) and float(value) > 0:
            active.append(col)
    return active


class MultiLabelTextDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        label_to_idx: Dict[str, int],
        text_col: str = "text",
        labels_col: str = "labels",
        label_sep: str = "|",
        max_length: int = 320,
        label_columns: Optional[List[str]] = None,
    ):
        self.texts = dataframe[text_col].fillna("").astype(str).tolist()
        if label_columns is not None:
            self.label_lists = [
                labels_from_row(row, label_columns)
                for _, row in dataframe.iterrows()
            ]
        else:
            self.label_lists = [
                split_labels(x, label_sep) for x in dataframe[labels_col].tolist()
            ]
        self.tokenizer = tokenizer
        self.label_to_idx = label_to_idx
        self.num_labels = len(label_to_idx)
        self.max_length = max_length

        # The paper uses one dedicated unused token per label if label descriptions
        # are unavailable. BERT-base-uncased contains [unused0], [unused1], ...
        vocab = tokenizer.get_vocab()
        self.label_token_ids = []
        for i in range(self.num_labels):
            token = f"[unused{i}]"
            if token not in vocab:
                raise ValueError(
                    f"{token} is not in this tokenizer vocabulary. "
                    "Use bert-base-uncased or implement a label-token alternative."
                )
            self.label_token_ids.append(vocab[token])

        if self.num_labels + 3 >= max_length:
            raise ValueError("max_length is too short for all label tokens and special tokens.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        tokens = self.tokenizer.tokenize(text)

        # Sequence: [CLS] text [SEP] label_1 ... label_L [SEP]
        # Reserve L + 3 positions for special/label tokens.
        text_budget = self.max_length - self.num_labels - 3
        tokens = tokens[:text_budget]
        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        input_ids = [cls_id] + text_ids + [sep_id] + self.label_token_ids + [sep_id]
        text_len = len(text_ids)

        # BERT token type 0 for [CLS] text [SEP], 1 for label tokens [SEP].
        token_type_ids = [0] * (text_len + 2) + [1] * (self.num_labels + 1)
        attention_mask = [1] * len(input_ids)

        # Pad to fixed max length so text-token positions are easy to batch.
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len

        target = torch.zeros(self.num_labels, dtype=torch.float32)
        for label in self.label_lists[index]:
            if label in self.label_to_idx:
                target[self.label_to_idx[label]] = 1.0

        # The text positions start at 1 and end at text_len inclusive.
        # [CLS] is not part of Hx. Text padding is excluded by text_mask.
        text_mask = torch.zeros(self.max_length, dtype=torch.bool)
        if text_len:
            text_mask[1:1 + text_len] = True

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "text_mask": text_mask,
            "target": target,
        }


class CORE(nn.Module):
    """
    CORE architecture:
      1) BERT jointly encodes text tokens and all label tokens.
      2) L2-normalized Hx and Hy form word-label correlation G = Hx @ Hy^T.
      3) A shared local convolution over token windows generates higher-order
         correlations; a per-label bias is learned.
      4) Max over labels -> softmax over valid text positions -> tanh.
      5) Attention-weighted sum of Hx -> sigmoid multi-label classifier.
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        window_size: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size must be positive.")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.num_labels = num_labels
        self.window_size = window_size

        # Shared convolution kernel W1 in R^window_size and per-label bias b1.
        self.correlation_kernel = nn.Parameter(torch.empty(window_size))
        nn.init.xavier_uniform_(self.correlation_kernel.unsqueeze(0))
        self.correlation_bias = nn.Parameter(torch.zeros(num_labels))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask, text_mask, target=None):
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state  # [B, sequence, hidden]

        # Find label token positions: labels are the L positions immediately before
        # the final [SEP], and the sequence is padded to max_length.
        # The first final-separator position is where token_type_ids changes back
        # to 0 only in padding; robustly use the known fixed layout:
        # label tokens are the last num_labels active positions before final [SEP].
        batch_size = input_ids.size(0)
        label_positions = []
        text_positions = []
        # Inputs are constructed with all labels before final [SEP], and pad after.
        # Use token_type_ids==1 and attention_mask==1; the last active position is SEP.
        for b in range(batch_size):
            type1_positions = torch.nonzero(
                (token_type_ids[b] == 1) & (attention_mask[b] == 1),
                as_tuple=False
            ).squeeze(-1)
            # type-1 positions include all labels plus trailing [SEP].
            label_positions.append(type1_positions[:-1])
            text_positions.append(torch.nonzero(text_mask[b], as_tuple=False).squeeze(-1))

        # Fixed label count; gather into [B, L, H].
        label_pos = torch.stack([p[:self.num_labels] for p in label_positions], dim=0)
        batch_idx = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
        Hy = hidden_states[batch_idx, label_pos]
        # Hx is variable-length, but padded into [B, max_text_tokens, H].
        max_text_tokens = max((len(p) for p in text_positions), default=1)
        max_text_tokens = max(max_text_tokens, 1)
        hx_list, valid_list = [], []
        for b, pos in enumerate(text_positions):
            hx = hidden_states[b, pos]
            n = hx.size(0)
            if n < max_text_tokens:
                hx = torch.cat(
                    [hx, hx.new_zeros(max_text_tokens - n, hx.size(-1))], dim=0
                )
            hx_list.append(hx)
            valid = torch.zeros(max_text_tokens, dtype=torch.bool, device=input_ids.device)
            valid[:n] = True
            valid_list.append(valid)
        Hx = torch.stack(hx_list, dim=0)               # [B, T, H]
        valid_text = torch.stack(valid_list, dim=0)    # [B, T]

        # Paper normalizes Hx and Hy by L2 norm before matrix multiplication.
        Hx_norm = F.normalize(Hx, p=2, dim=-1, eps=1e-12)
        Hy_norm = F.normalize(Hy, p=2, dim=-1, eps=1e-12)
        G = torch.matmul(Hx_norm, Hy_norm.transpose(1, 2))  # [B, T, L]

        # Local windows over the token axis. Padding is symmetric ("same") so
        # every text token receives one correlation vector. For even kernels,
        # left/right padding differs by one position.
        k = self.window_size
        total_pad = k - 1
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        G_padded = F.pad(G.transpose(1, 2), (pad_left, pad_right))  # [B, L, T+K-1]
        # Unfold gives [B, L, T, K].
        windows = G_padded.unfold(dimension=2, size=k, step=1).permute(0, 2, 3, 1)
        # [B, T, K, L] * [K] -> [B, T, L], plus per-label bias and ReLU.
        M = F.relu(
            (windows * self.correlation_kernel.view(1, 1, k, 1)).sum(dim=2)
            + self.correlation_bias.view(1, 1, self.num_labels)
        )

        # Max over labels, then softmax over valid text positions.
        token_scores = M.max(dim=-1).values  # [B, T]
        token_scores = token_scores.masked_fill(~valid_text, torch.finfo(token_scores.dtype).min)
        alpha = torch.softmax(token_scores, dim=-1)
        alpha = torch.tanh(alpha)  # paper applies softmax and tanh sequentially
        alpha = alpha * valid_text.to(alpha.dtype)

        # c = alpha^T Hx
        context = torch.bmm(alpha.unsqueeze(1), Hx).squeeze(1)
        logits = self.classifier(self.dropout(context))
        loss = None
        if target is not None:
            loss = F.binary_cross_entropy_with_logits(logits, target)
        return {"loss": loss, "logits": logits, "attention": alpha}


def compute_metrics(y_true, probabilities, threshold=0.5):
    y_pred = (probabilities >= threshold).astype(np.int32)
    metrics = {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    # Ranking metrics require at least one positive and one negative label per row
    # and may be undefined for degenerate small samples.
    try:
        metrics["macro_auc"] = float(roc_auc_score(y_true, probabilities, average="macro"))
    except ValueError:
        metrics["macro_auc"] = None
    try:
        metrics["ranking_loss"] = float(label_ranking_loss(y_true, probabilities))
    except ValueError:
        metrics["ranking_loss"] = None
    try:
        metrics["coverage"] = float(coverage_error(y_true, probabilities) / y_true.shape[1])
    except ValueError:
        metrics["coverage"] = None
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    losses, all_targets, all_probs = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            text_mask=batch["text_mask"],
            target=batch["target"],
        )
        losses.append(out["loss"].item())
        all_targets.append(batch["target"].cpu().numpy())
        all_probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
    y_true = np.concatenate(all_targets, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    metrics = compute_metrics(y_true, probs, threshold=threshold)
    metrics["loss"] = float(np.mean(losses))
    return metrics, y_true, probs


def make_label_vocab(train_df, labels_col, label_sep, text_col="text"):
    # Native AAPD format: every non-text column is a binary label column.
    if labels_col not in train_df.columns:
        label_columns = [c for c in train_df.columns if c != text_col]
        if not label_columns:
            raise ValueError("No label columns found in native multi-hot dataset.")
        return {label: i for i, label in enumerate(label_columns)}, label_columns
    label_set = set()
    for value in train_df[labels_col].tolist():
        label_set.update(split_labels(value, label_sep))
    return {label: i for i, label in enumerate(sorted(label_set))}, None


def train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_csv(args.train)
    valid_df = pd.read_csv(args.valid)
    test_df = pd.read_csv(args.test)
    for df_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        if args.text_col not in df:
            raise ValueError(
                f"{df_name} CSV must contain text column '{args.text_col}'. "
                f"Available columns: {list(df.columns)[:8]} ..."
            )

    label_to_idx, label_columns = make_label_vocab(
        train_df, args.labels_col, args.label_sep, text_col=args.text_col
    )
    if label_columns is not None:
        # Ensure all splits have exactly the same label columns in the same order.
        for split_name, df in [("valid", valid_df), ("test", test_df)]:
            missing = [c for c in label_columns if c not in df.columns]
            if missing:
                raise ValueError(f"{split_name} split is missing label columns: {missing}")
        print("Detected native AAPD multi-hot format (abstract + binary label columns).")
    else:
        print("Detected converted text + pipe-separated labels format.")
    with open(os.path.join(args.output_dir, "label_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)
    print(f"Number of labels from train split: {len(label_to_idx)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = MultiLabelTextDataset(
        train_df, tokenizer, label_to_idx, args.text_col, args.labels_col,
        args.label_sep, args.max_length, label_columns
    )
    valid_ds = MultiLabelTextDataset(
        valid_df, tokenizer, label_to_idx, args.text_col, args.labels_col,
        args.label_sep, args.max_length, label_columns
    )
    test_ds = MultiLabelTextDataset(
        test_df, tokenizer, label_to_idx, args.text_col, args.labels_col,
        args.label_sep, args.max_length, label_columns
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CORE(args.model_name, len(label_to_idx), args.window_size, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_valid_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch["input_ids"],
                token_type_ids=batch["token_type_ids"],
                attention_mask=batch["attention_mask"],
                text_mask=batch["text_mask"],
                target=batch["target"],
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_losses.append(out["loss"].item())
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={np.mean(train_losses):.4f}")

        valid_metrics, _, _ = evaluate(model, valid_loader, device, args.threshold)
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **{
            f"valid_{k}": v for k, v in valid_metrics.items()
        }}
        history.append(record)
        print(json.dumps(record, indent=2))
        if valid_metrics["micro_f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["micro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model_name,
                "num_labels": len(label_to_idx),
                "window_size": args.window_size,
                "max_length": args.max_length,
                "label_to_idx": label_to_idx,
                "args": vars(args),
            }, os.path.join(args.output_dir, "best_model.pt"))

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, y_true, probabilities = evaluate(model, test_loader, device, args.threshold)
    print("TEST METRICS")
    print(json.dumps(test_metrics, indent=2))
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    np.save(os.path.join(args.output_dir, "test_probabilities.npy"), probabilities)
    np.save(os.path.join(args.output_dir, "test_targets.npy"), y_true)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--text-col", default="abstract")
    parser.add_argument("--labels-col", default="labels")
    parser.add_argument("--label-sep", default="|")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", default="runs/core")
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    sys.argv = [
        "CORE_implementation.py",
        "--train", "AAPD/train.csv",
        "--valid", "AAPD/dev.csv",
        "--test", "AAPD/test.csv",
        "--text-col", "abstract",
        "--model-name", "bert-base-uncased",
        "--max-length", "320",
        "--window-size", "10",
        "--batch-size", "32",
        "--learning-rate", "5e-5",
        "--epochs", "5",
        "--output-dir", "runs/core_aapd"
    ]
    
    train(parse_args())