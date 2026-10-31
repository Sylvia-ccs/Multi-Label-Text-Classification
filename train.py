import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from core.data import build_datasets
from core.model import CORE
from core.metrics import calculate_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                text_lengths=batch["text_lengths"],
                labels=batch["labels"],
            )
            probs = torch.sigmoid(out["logits"])
            all_y.append(batch["labels"].cpu().numpy())
            all_p.append(probs.cpu().numpy())
            n = batch["labels"].size(0)
            total_loss += out["loss"].item() * n
            total_n += n
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    metrics = calculate_metrics(y, p)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train.src/.tgt, valid.src/.tgt, test.src/.tgt and topic_sorted.json")
    parser.add_argument("--output_dir", type=str, default="outputs/rcv1_core")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_length", type=int, default=320)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    datasets, label_names, collator = build_datasets(
        data_dir, tokenizer, args.max_seq_length
    )
    model = CORE(
        pretrained_model=args.model_name,
        num_labels=len(label_names),
        window_size=args.window_size,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        datasets["valid"], batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_micro_f1 = -1.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    text_lengths=batch["text_lengths"],
                    labels=batch["labels"],
                )
                loss = out["loss"] / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.gradient_accumulation_steps

            should_step = (
                (step + 1) % args.gradient_accumulation_steps == 0
                or (step + 1) == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

        valid_metrics = evaluate(model, valid_loader, device)
        print(f"\nEpoch {epoch} validation:", json.dumps(valid_metrics, indent=2))
        if valid_metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = valid_metrics["micro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_names": label_names,
                "args": vars(args),
            }, best_path)
            print("Saved best checkpoint:", best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "label_names.json").write_text(
        json.dumps(label_names, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
