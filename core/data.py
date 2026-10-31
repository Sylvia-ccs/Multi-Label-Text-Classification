import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset


def load_label_names(data_dir: Path, train_targets: List[List[str]]) -> List[str]:
    """Read the supplied topic order when possible; otherwise derive a stable order."""
    for filename in ("topic_sorted.json", "label_to_id.json"):
        path = data_dir / filename
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            names = [str(x) for x in obj]
            if names and len(names) >= 3 and names[0].lower() in {"<blank>", "<unk>", "<s>"}:
                names = names[3:]
            return names
        if isinstance(obj, dict):
            # label -> integer id
            if all(isinstance(v, int) for v in obj.values()):
                pairs = sorted(obj.items(), key=lambda kv: kv[1])
                names = [str(k) for k, _ in pairs]
                names = [x for x in names if x not in {"<blank>", "<unk>", "<s>"}]
                return names
            # integer id -> label
            if all(str(k).isdigit() for k in obj):
                pairs = sorted(obj.items(), key=lambda kv: int(kv[0]))
                names = [str(v) for _, v in pairs]
                names = [x for x in names if x not in {"<blank>", "<unk>", "<s>"}]
                return names

    return sorted({label for row in train_targets for label in row})


def read_split(data_dir: Path, split: str):
    src_path = data_dir / f"{split}.src"
    tgt_path = data_dir / f"{split}.tgt"
    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(
            f"Expected {src_path} and {tgt_path}. Extract the RCV1-V2 archive first."
        )
    texts = src_path.read_text(encoding="utf-8").splitlines()
    targets = [line.strip().split() for line in tgt_path.read_text(encoding="utf-8").splitlines()]
    if len(texts) != len(targets):
        raise ValueError(f"{split}: {len(texts)} text rows but {len(targets)} label rows")
    return texts, targets


class MultiLabelTextDataset(Dataset):
    def __init__(self, texts, targets, label_to_id, tokenizer, max_text_tokens):
        self.examples = []
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_text_tokens = max_text_tokens

        for text, labels in zip(texts, targets):
            # The RCV1 source files are already tokenized/stemmed strings.
            text_ids = tokenizer.encode(
                text, add_special_tokens=False, truncation=True,
                max_length=max_text_tokens
            )
            target = torch.zeros(len(label_to_id), dtype=torch.float32)
            for label in labels:
                if label in label_to_id:
                    target[label_to_id[label]] = 1.0
            self.examples.append((text_ids, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CoreCollator:
    """Build [CLS] text [SEP] label_1 ... label_L [SEP] sequences."""
    def __init__(self, label_token_ids, pad_token_id):
        self.label_token_ids = label_token_ids
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch_size = len(batch)
        n_labels = len(self.label_token_ids)
        max_text_len = max(max(1, len(x[0])) for x in batch)

        input_rows, attention_rows, text_lengths, targets = [], [], [], []
        for text_ids, target in batch:
            # Empty text is represented by one [UNK] token to keep attention pooling defined.
            if not text_ids:
                text_ids = [self.pad_token_id]
            text_ids = text_ids[:max_text_len]
            text_len = len(text_ids)
            row = [self.cls_id] + text_ids + [self.sep_id] + self.label_token_ids + [self.sep_id]
            input_rows.append(row)
            attention_rows.append([1] * len(row))
            text_lengths.append(text_len)
            targets.append(target)

        max_len = max(len(x) for x in input_rows)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, (ids, mask) in enumerate(zip(input_rows, attention_rows)):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(mask)] = torch.tensor(mask, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_lengths": torch.tensor(text_lengths, dtype=torch.long),
            "labels": torch.stack(targets),
        }

    @classmethod
    def with_tokenizer(cls, label_token_ids, tokenizer):
        obj = cls(label_token_ids, tokenizer.pad_token_id)
        obj.cls_id = tokenizer.cls_token_id
        obj.sep_id = tokenizer.sep_token_id
        return obj


def build_datasets(data_dir, tokenizer, max_seq_length):
    train_texts, train_targets = read_split(data_dir, "train")
    label_names = load_label_names(data_dir, train_targets)
    if len(label_names) != 103:
        raise ValueError(
            f"Expected 103 RCV1-V2 topics, found {len(label_names)}. "
            "Check topic_sorted.json / label_to_id.json and target files."
        )
    label_to_id = {name: i for i, name in enumerate(label_names)}

    # Sequence = [CLS] + text + [SEP] + 103 label tokens + [SEP].
    max_text_tokens = max_seq_length - len(label_names) - 3
    if max_text_tokens < 8:
        raise ValueError("max_seq_length is too small for 103 label tokens.")

    valid_texts, valid_targets = read_split(data_dir, "valid")
    test_texts, test_targets = read_split(data_dir, "test")

    datasets = {
        "train": MultiLabelTextDataset(train_texts, train_targets, label_to_id, tokenizer, max_text_tokens),
        "valid": MultiLabelTextDataset(valid_texts, valid_targets, label_to_id, tokenizer, max_text_tokens),
        "test": MultiLabelTextDataset(test_texts, test_targets, label_to_id, tokenizer, max_text_tokens),
    }

    # Use existing BERT unused tokens first, then add new tokens if needed.
    existing = [f"[unused{i}]" for i in range(100)]
    label_tokens = existing[:min(100, len(label_names))]
    if len(label_names) > 100:
        label_tokens += [f"[CORE_LABEL_{i}]" for i in range(100, len(label_names))]
        tokenizer.add_tokens(label_tokens[100:])
    label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
    if any(x == tokenizer.unk_token_id for x in label_token_ids):
        raise ValueError("A label token was mapped to [UNK]; check tokenizer vocabulary.")
    collator = CoreCollator.with_tokenizer(label_token_ids, tokenizer)
    return datasets, label_names, collator
