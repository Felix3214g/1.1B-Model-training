#!/usr/bin/env python
import argparse
import os
import sys
import subprocess
import logging
import math
from pathlib import Path
from typing import List

REQUIRED_PKGS = {
    "transformers": ">=4.34.0",
    "datasets": ">=2.16.0",
    "accelerate": ">=0.25.0",
    "tokenizers": ">=0.15.0",
    "tqdm": ">=4.66.0",
}

def _ensure_deps():
    import importlib
    to_install: List[str] = []
    for pkg, spec in REQUIRED_PKGS.items():
        try:
            importlib.import_module(pkg)
        except ImportError:
            to_install.append(f"{pkg}{spec}")
    if to_install:
        print("[Setup] Installiere fehlende Pakete: ", " ".join(to_install))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])

_ensure_deps()

import torch
from torch.utils.data import IterableDataset
import datasets
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
from accelerate import Accelerator
from tqdm.auto import tqdm

SPECIAL_TOKENS = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end": "<|endoftext|>",
    "pad": "<|pad|>",
}

MODEL_CONFIG = {
    "n_layer": 36,
    "n_head": 16,
    "n_embd": 1600,
    "n_positions": 2048,
    "n_ctx": 2048,
}

def parse_args():
    p = argparse.ArgumentParser(description="Train / Resume / Infer Chat‑GPT‑like model")
    p.add_argument("--mode", choices=["train", "resume", "inference"], default="train")
    p.add_argument("--dataset_names", nargs="+", default=[
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "OpenAssistant/oasst1",
    ])
    p.add_argument("--output_dir", type=str, default="./chatbot_output")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--context_length", type=int, default=2048)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--streaming", action="store_true", help="Use HF streaming datasets")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inference_model_path", type=str)
    p.add_argument("--max_new_tokens", type=int, default=200)
    return p.parse_args()

def format_conversation(example):
    conv_key = None
    for ck in ("conversations", "conversation"):
        if ck in example and isinstance(example[ck], list):
            conv_key = ck
            break
    if conv_key:
        text = "\n".join(
            [
                f"{SPECIAL_TOKENS['user']} {turn.get('value', turn.get('text', ''))}"
                if turn.get("from", turn.get("role", "")).lower() in {"human", "user", "prompter"}
                else f"{SPECIAL_TOKENS['assistant']} {turn.get('value', turn.get('text', ''))}"
                for turn in example[conv_key]
                if isinstance(turn, dict)
            ]
        )
    elif "messages" in example:
        parts = []
        for msg in example["messages"]:
            if msg.get("role") == "prompter":
                parts.append(f"{SPECIAL_TOKENS['user']} {msg['text']}")
            elif msg.get("role") == "assistant":
                parts.append(f"{SPECIAL_TOKENS['assistant']} {msg['text']}")
        text = "\n".join(parts)
    elif {"prompt", "completion"}.issubset(example.keys()):
        text = (
            f"{SPECIAL_TOKENS['user']} {example['prompt']}\n"
            f"{SPECIAL_TOKENS['assistant']} {example['completion']}"
        )
    elif "text" in example and isinstance(example["text"], str):
        text = example["text"]
    else:
        return {"text": ""}
    return {"text": text + f"\n{SPECIAL_TOKENS['end']}"}

class StreamingChatDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, block_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        # store block size explicitly to avoid confusion
        self.block_size = block_size

    def __iter__(self):
        for ex in self.dataset:
            formatted = format_conversation(ex)["text"]
            if not formatted:
                continue
            tokens = self.tokenizer(formatted)["input_ids"]
            for i in range(0, len(tokens), self.block_size):
                chunk = tokens[i : i + self.block_size]
                if len(chunk) < self.block_size:
                    continue
                yield {
                    "input_ids": chunk,
                    "labels": chunk.copy(),
                    "attention_mask": [1] * self.block_size,
                }

def get_or_build_tokenizer(out_dir: Path):
    if (out_dir / "tokenizer").exists():
        tok = AutoTokenizer.from_pretrained(out_dir / "tokenizer")
    else:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        tok.add_special_tokens(
            {
                "pad_token": SPECIAL_TOKENS["pad"],
                "bos_token": SPECIAL_TOKENS["end"],
                "eos_token": SPECIAL_TOKENS["end"],
                "additional_special_tokens": [
                    SPECIAL_TOKENS["user"],
                    SPECIAL_TOKENS["assistant"],
                ],
            }
        )
        tok.save_pretrained(out_dir / "tokenizer")
    return tok

def build_model(tokenizer, context_len):
    cfg_dict = dict(MODEL_CONFIG)
    cfg_dict.update(
        {
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "n_ctx": context_len,
            "n_positions": context_len,
        }
    )
    config = GPT2Config(**cfg_dict)
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    return model

def main():
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("chatbot_train")
    set_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    tokenizer = get_or_build_tokenizer(out_dir)
    block_size = min(args.context_length, tokenizer.model_max_length)
    if args.mode in {"train", "resume"}:
        logger.info("Lade Datensätze …")
        raw_sets = []
        for name in args.dataset_names:
            ds = None
            for stream_flag in ([args.streaming] + ([True] if not args.streaming else [])):
                try:
                    ds = load_dataset(name, split="train", streaming=stream_flag, trust_remote_code=True)
                    if stream_flag and not args.streaming:
                        logger.warning("'%s' wurde erfolgreich im Streaming‑Modus geladen.", name)
                    break
                except datasets.exceptions.DataFilesNotFoundError:
                    logger.warning("Keine Daten für '%s' mit streaming=%s gefunden.", name, stream_flag)
                except Exception as e:
                    logger.error("Fehler beim Laden von '%s' (streaming=%s): %s", name, stream_flag, e)
                    break
            if ds is None:
                logger.error("Überspringe Datensatz '%s' – konnte nicht geladen werden.", name)
                continue
            raw_sets.append(ds)
        if not raw_sets:
            raise RuntimeError("Kein Datensatz konnte geladen werden. Bitte Dataset‑Namen prüfen oder --streaming aktivieren.")
        if args.streaming:
            combined = datasets.interleave_datasets(raw_sets)
            train_data = StreamingChatDataset(combined, tokenizer, block_size)
            eval_data = None
        else:
            train_sets = []
            for ds in raw_sets:
                f_ds = ds.map(format_conversation, remove_columns=ds.column_names, desc="format")
                f_ds = f_ds.filter(lambda x: bool(x["text"]))
                try:
                    if len(f_ds) == 0:
                        logger.warning("Datensatz '%s' enthält nach Formatierung keine gültigen Beispiele – überspringe.", ds.builder_name if hasattr(ds, 'builder_name') else 'unknown')
                        continue
                except TypeError:
                    pass
                train_sets.append(f_ds)
            if not train_sets:
                raise RuntimeError("Nach Formatierung/Filterung sind keine Trainingsbeispiele übrig geblieben. Prüfe Datensätze oder aktiviere --streaming.")
            merged: Dataset = datasets.concatenate_datasets(train_sets).shuffle(seed=args.seed)
            def tokenize(batch):
                ids = tokenizer(batch["text"], truncation=True, max_length=block_size)
                batch["input_ids"] = ids["input_ids"]
                batch["attention_mask"] = ids["attention_mask"]
                batch["labels"] = ids["input_ids"]
                return batch
            train_data = merged.map(tokenize, batched=True, remove_columns=["text"], num_proc=os.cpu_count())
            eval_data = None
            if not train_data:
                raise ValueError("Trainingsdatensatz ist nach der Verarbeitung leer. Überprüfe Filter oder Formatierung.")
            required_cols = {"input_ids", "attention_mask", "labels"}
            if not required_cols.issubset(train_data.column_names):
                raise ValueError(f"Erwartete Spalten {required_cols} nicht im Trainingsdatensatz gefunden. Vorhanden: {train_data.column_names}")
        model = build_model(tokenizer, block_size)
        n_params = count_trainable_parameters(model)
        logger.info("Trainable parameters: %.1f M", n_params / 1e6)
        print(f"{n_params/1e6:.1f} M trainable parameters")
        training_args = TrainingArguments(
            output_dir=str(out_dir),
            overwrite_output_dir=args.mode == "train",
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            save_steps=500,
            logging_steps=50,
            evaluation_strategy="no",
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            report_to="none",
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        logger.info("Starte Training …")
        try:
            trainer.train(resume_from_checkpoint=args.mode == "resume")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("OOM aufgetreten – reduziere Kontextlänge auf 1024 und versuche erneut …")
                trainer.args.per_device_train_batch_size = 1
                trainer.args.gradient_accumulation_steps *= 2
                trainer.model.config.n_ctx = 1024
                trainer.model.config.n_positions = 1024
                trainer.train()
            else:
                raise
        logger.info("Training beendet. Speichere finalen Checkpoint …")
        trainer.save_model(out_dir / "final_model")
        tokenizer.save_pretrained(out_dir / "final_model")
    elif args.mode == "inference":
        model_path = Path(args.inference_model_path or out_dir / "final_model")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
        model.eval()
        stop_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS.values() if tok in tokenizer.vocab]
        print("Chatbot bereit – tippe 'quit' zum Beenden.")
        history = ""
        while True:
            prompt = input(f"{SPECIAL_TOKENS['user']} ")
            if prompt.lower() in {"quit", "exit"}:
                break
            history += f"{SPECIAL_TOKENS['user']} {prompt}\n{SPECIAL_TOKENS['assistant']}"
            inputs = tokenizer(history, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=stop_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )
            answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            print(f"{SPECIAL_TOKENS['assistant']} {answer}\n")
            history += " " + answer + f"\n{SPECIAL_TOKENS['end']}\n"
    else:
        raise ValueError("Unbekannter Modus")

def count_trainable_parameters(model: 'torch.nn.Module') -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    main()
