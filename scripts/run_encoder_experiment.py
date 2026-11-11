#!/usr/bin/env python3
"""Run one reproducible encoder experiment and save per-example predictions.

The runner is intentionally script-based rather than notebook-state-based. It
supports full fine-tuning and a genuinely frozen encoder, batches evaluation,
selects the best epoch on the Turkish development split, and records enough
metadata to trace every result.

A local random tiny model is available only for pipeline smoke tests:

    python scripts/run_encoder_experiment.py --smoke-test

Smoke-test metrics are explicitly marked as scientifically invalid.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)


VIEW_COLUMNS = {
    "surface": "text",
    "lemma": "lemma_text",
    "morph": "morph_text",
    "randomized_morph": "randomized_morph_text",
}


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    view: str
    seed: int
    max_length: int
    epochs: int
    learning_rate: float
    weight_decay: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    frozen_encoder: bool
    precision: str
    smoke_test: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_standardized(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("--data must be CSV or Parquet")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


class EncodedTextDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str,
        max_length: int,
    ) -> None:
        self.metadata = frame[
            [
                "example_id",
                "language",
                "label",
                "split",
                "source_dataset",
                "text_sha256",
            ]
        ].reset_index(drop=True)
        texts = frame[text_column].fillna("").astype(str).tolist()
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def load_model_and_tokenizer(args: argparse.Namespace):
    if args.smoke_test:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(args.local_tokenizer),
            bos_token="<s>",
            eos_token="</s>",
            cls_token="<s>",
            sep_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            mask_token="<mask>",
        )
        config = XLMRobertaConfig(
            vocab_size=len(tokenizer),
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=max(args.max_length + 2, 130),
            num_labels=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        model = XLMRobertaForSequenceClassification(config)
        return model, tokenizer, "LOCAL_RANDOM_TINY_XLMR"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        num_labels=2,
    )
    return model, tokenizer, args.model_name


def freeze_base_encoder(model: torch.nn.Module) -> None:
    prefix = getattr(model, "base_model_prefix", None)
    base = getattr(model, prefix, None) if prefix else None
    if base is None:
        base = getattr(model, "base_model", None)
    if base is None:
        raise ValueError("Could not locate the model's base encoder to freeze")
    for parameter in base.parameters():
        parameter.requires_grad = False


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


def make_loader(
    dataset: EncodedTextDataset,
    *,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def resolve_precision(requested: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return requested


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_abusive": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    losses: list[float] = []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast_context(device, precision):
            outputs = model(**batch)
        if outputs.loss is not None:
            losses.append(float(outputs.loss.detach().cpu()))
        probs = torch.softmax(outputs.logits.float(), dim=-1)
        labels.extend(batch["labels"].detach().cpu().tolist())
        predictions.extend(torch.argmax(probs, dim=-1).detach().cpu().tolist())
        probabilities.extend(probs[:, 1].detach().cpu().tolist())
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return (
        np.asarray(labels, dtype=int),
        np.asarray(predictions, dtype=int),
        np.asarray(probabilities, dtype=float),
        mean_loss,
    )


def save_checkpoint(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)


def restore_checkpoint(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    from safetensors.torch import load_file

    state_path = checkpoint_dir / "model.safetensors"
    if not state_path.exists():
        # Transformers may shard large models.
        index_path = checkpoint_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No safe checkpoint found in {checkpoint_dir}")
        index = json.loads(index_path.read_text())
        state: dict[str, torch.Tensor] = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(checkpoint_dir / shard), device="cpu"))
    else:
        state = load_file(str(state_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> list[dict[str, Any]]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_updates = max(updates_per_epoch * args.epochs, 1)
    warmup_steps = int(round(total_updates * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and precision == "fp16")
    )
    best_macro_f1 = -1.0
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer_steps = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            with autocast_context(device, precision):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            should_step = (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        labels, predictions, _, dev_loss = predict(
            model, dev_loader, device=device, precision=precision
        )
        metrics = binary_metrics(labels, predictions)
        epoch_record = {
            "epoch": epoch,
            "mean_train_batch_loss": running_loss / max(len(train_loader), 1),
            "optimizer_steps": optimizer_steps,
            "dev_loss": dev_loss,
            **metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} train_loss={epoch_record['mean_train_batch_loss']:.4f} "
            f"dev_macro_f1={metrics['macro_f1']:.4f}"
        )
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            save_checkpoint(model, checkpoint_dir)

    restore_checkpoint(model, checkpoint_dir)
    return history


def frame_for_primary_experiment(frame: pd.DataFrame) -> pd.DataFrame:
    # The paired Finnish diagnostic is intentionally excluded from primary
    # language-level metrics and evaluated separately by a dedicated analysis.
    return frame.loc[~frame["source_dataset"].eq("RTP-LX-FI-PAIRED")].copy()


def select_training_frames(
    frame: pd.DataFrame, *, smoke_test: bool, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = frame.loc[
        frame["language"].eq("TR") & frame["source_dataset"].eq("ATC")
    ].copy()
    train_frame = source.loc[source["split"].eq("train")].copy()
    dev_frame = source.loc[source["split"].eq("dev")].copy()
    if smoke_test:
        train_frame = stratified_small_sample(train_frame, total=24, seed=seed)
        dev_frame = stratified_small_sample(dev_frame, total=16, seed=seed)
    if train_frame.empty or dev_frame.empty:
        raise ValueError("Turkish ATC train/dev rows are required")
    return train_frame, dev_frame


def stratified_small_sample(frame: pd.DataFrame, *, total: int, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    labels = sorted(frame["label"].unique())
    per_label = total // len(labels)
    for label in labels:
        group = frame.loc[frame["label"].eq(label)]
        parts.append(group.sample(n=min(per_label, len(group)), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def evaluate_and_write(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    frame: pd.DataFrame,
    *,
    text_column: str,
    device: torch.device,
    precision: str,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    eval_frame = frame.loc[frame["split"].eq("test")].copy()
    if args.smoke_test:
        sampled: list[pd.DataFrame] = []
        for language, group in eval_frame.groupby("language", sort=True):
            sampled.append(stratified_small_sample(group, total=12, seed=args.seed))
        eval_frame = pd.concat(sampled, ignore_index=True)

    for (language, dataset_name), group in eval_frame.groupby(
        ["language", "source_dataset"], sort=True
    ):
        dataset = EncodedTextDataset(
            group,
            tokenizer=tokenizer,
            text_column=text_column,
            max_length=args.max_length,
        )
        loader = make_loader(
            dataset,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
        )
        labels, predictions, probabilities, loss = predict(
            model, loader, device=device, precision=precision
        )
        metrics = binary_metrics(labels, predictions)
        metric_rows.append(
            {
                "run_id": run_id,
                "model": args.model_name,
                "view": args.view,
                "seed": args.seed,
                "max_length": args.max_length,
                "frozen_encoder": args.frozen_encoder,
                "language": language,
                "source_dataset": dataset_name,
                "split": "test",
                "n": len(group),
                "loss": loss,
                **metrics,
                "scientifically_valid": not args.smoke_test,
            }
        )
        output = dataset.metadata.copy()
        output["prediction"] = predictions
        output["probability_1"] = probabilities
        output["run_id"] = run_id
        output["model"] = args.model_name
        output["view"] = args.view
        output["seed"] = args.seed
        output["max_length"] = args.max_length
        output["frozen_encoder"] = args.frozen_encoder
        output["scientifically_valid"] = not args.smoke_test
        prediction_parts.append(output)

    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_csv(
        run_dir / "predictions.csv", index=False
    )
    return metric_rows


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/standardized_all.csv"))
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--view", choices=sorted(VIEW_COLUMNS), default="surface")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto"
    )
    parser.add_argument("--frozen-encoder", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/encoder"))
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--local-tokenizer", type=Path, default=Path("tokenizer.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length < 8:
        raise ValueError("--max-length must be at least 8")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")

    set_seed(args.seed)
    frame = read_standardized(args.data)
    required = {
        "example_id",
        "language",
        "text",
        "label",
        "split",
        "source_dataset",
        "text_sha256",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Standardized data is missing columns: {sorted(missing)}")
    text_column = VIEW_COLUMNS[args.view]
    if text_column not in frame.columns:
        raise ValueError(
            f"View {args.view!r} needs column {text_column!r}. "
            "Run scripts/build_ud_morph_views.py first."
        )
    frame = frame_for_primary_experiment(frame)
    train_frame, dev_frame = select_training_frames(
        frame, smoke_test=args.smoke_test, seed=args.seed
    )

    model, tokenizer, effective_model_name = load_model_and_tokenizer(args)
    if args.frozen_encoder:
        freeze_base_encoder(model)
    counts = parameter_counts(model)
    if args.frozen_encoder and counts["frozen"] == 0:
        raise AssertionError("Frozen baseline has no frozen parameters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_precision = resolve_precision(args.precision, device)
    if not args.smoke_test and device.type != "cuda":
        raise RuntimeError(
            "Publication experiments require a CUDA GPU. Use --smoke-test only "
            "to validate the pipeline on CPU."
        )
    model.to(device)

    run_id = (
        f"{slug(effective_model_name)}__{args.view}__seed{args.seed}__L{args.max_length}"
        + ("__frozen" if args.frozen_encoder else "__full")
        + ("__SMOKE" if args.smoke_test else "")
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "best_checkpoint"

    config = RunConfig(
        model_name=effective_model_name,
        view=args.view,
        seed=args.seed,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        frozen_encoder=args.frozen_encoder,
        precision=effective_precision,
        smoke_test=args.smoke_test,
    )
    metadata = {
        "run_id": run_id,
        "config": asdict(config),
        "requested_model_name": args.model_name,
        "model_revision": args.model_revision,
        "data_path": str(args.data),
        "data_sha256": sha256_file(args.data),
        "train_rows": int(len(train_frame)),
        "dev_rows": int(len(dev_frame)),
        "parameter_counts": counts,
        "device": str(device),
        "requested_precision": args.precision,
        "effective_precision": effective_precision,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scientifically_valid": not args.smoke_test,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "started_unix": time.time(),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_dataset = EncodedTextDataset(
        train_frame,
        tokenizer=tokenizer,
        text_column=text_column,
        max_length=args.max_length,
    )
    dev_dataset = EncodedTextDataset(
        dev_frame,
        tokenizer=tokenizer,
        text_column=text_column,
        max_length=args.max_length,
    )
    train_loader = make_loader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=args.train_batch_size,
        shuffle=True,
        seed=args.seed,
    )
    dev_loader = make_loader(
        dev_dataset,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed,
    )

    history = train(
        model,
        train_loader,
        dev_loader,
        device=device,
        precision=effective_precision,
        args=args,
        checkpoint_dir=checkpoint_dir,
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = evaluate_and_write(
        model,
        tokenizer,
        frame,
        text_column=text_column,
        device=device,
        precision=effective_precision,
        args=args,
        run_id=run_id,
        run_dir=run_dir,
    )
    metadata["completed_unix"] = time.time()
    metadata["best_epoch"] = int(
        max(history, key=lambda item: item["macro_f1"])["epoch"]
    )
    metadata["metrics_rows"] = len(metrics)
    (run_dir / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.keep_checkpoint:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
