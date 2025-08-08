import argparse
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List
from itertools import chain

REQUIRED_PKGS = {
    "transformers": ">=4.34.0",
    "datasets": ">=2.16.0",
    "accelerate": ">=0.25.0",
    "tokenizers": ">=0.15.0",
    "tqdm": ">=4.66.0",
    "torch": ">=2.1.0",
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
        print("[Setup] Installing missing packages:", " ".join(to_install))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])

_ensure_deps()

import torch
from torch.utils.data import IterableDataset
import datasets
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
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
    "n_positions": 2048,  # context length target
}

def _gpu_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    # Ampere (8.0) or newer generally supports bfloat16
    return (major, minor) >= (8, 0)


def count_trainable_parameters(model: 'torch.nn.Module') -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    p = argparse.ArgumentParser(description="Train / Resume / Infer GPT‑2-style chat model")
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
            if msg.get("role") in ("prompter", "user"):
                parts.append(f"{SPECIAL_TOKENS['user']} {msg.get('text','')}")
            elif msg.get("role") == "assistant":
                parts.append(f"{SPECIAL_TOKENS['assistant']} {msg.get('text','')}")
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
        self.block_size = block_size

    def __iter__(self):
        for ex in self.dataset:
            formatted = format_conversation(ex)["text"]
            if not formatted.strip():
                continue
            tokens = self.tokenizer(formatted, truncation=True, max_length=self.block_size)["input_ids"]
            for i in range(0, len(tokens), self.block_size):
                chunk = tokens[i : i + self.block_size]
                if len(chunk) == self.block_size:
                    yield {
                        "input_ids": chunk,
                        "labels": chunk.copy(),
                        "attention_mask": [1] * self.block_size,
                    }


def get_or_build_tokenizer(out_dir: Path, context_len: int):
    tokenizer_path = out_dir / "tokenizer"
    if tokenizer_path.exists():
        tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    else:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        special_tokens_dict = {
            "pad_token": SPECIAL_TOKENS["pad"],
            "bos_token": SPECIAL_TOKENS["end"],  # keep EOS as BOS for GPT-2 compat
            "eos_token": SPECIAL_TOKENS["end"],
            "additional_special_tokens": [
                SPECIAL_TOKENS["user"],
                SPECIAL_TOKENS["assistant"],
            ],
        }
        tok.add_special_tokens(special_tokens_dict)
        tok.save_pretrained(tokenizer_path)
    # Ensure we can actually use the requested context length
    tok.model_max_length = context_len
    tok.padding_side = "left"  # safer for generation with causal models
    return tok


def build_model(tokenizer, context_len):
    cfg_dict = dict(MODEL_CONFIG)
    cfg_dict.update(
        {
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "n_positions": context_len,
        }
    )
    config = GPT2Config(**cfg_dict)
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    # enable gc to reduce memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def load_and_prepare_datasets(dataset_names, streaming, logger):
    raw_sets = []
    for name in dataset_names:
        try:
            ds = load_dataset(name, split="train", streaming=streaming, trust_remote_code=True)
            raw_sets.append(ds)
            logger.info(f"Loaded dataset: {name}")
        except Exception as e:
            logger.error(f"Failed to load dataset '{name}': {e}")
    if not raw_sets:
        raise RuntimeError("No datasets could be loaded. Check names or try --streaming.")
    return raw_sets


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("chatbot_train")

    set_seed(args.seed)

    # tokenizer & context length
    tokenizer = get_or_build_tokenizer(out_dir, args.context_length)
    block_size = args.context_length  # trust our config

    if args.mode in {"train", "resume"}:
        logger.info("Loading datasets...")
        raw_sets = load_and_prepare_datasets(args.dataset_names, args.streaming, logger)

        if args.streaming:
            combined = datasets.interleave_datasets(raw_sets)
            train_data = StreamingChatDataset(combined, tokenizer, block_size)
            eval_data = None
        else:
            train_sets = []
            for ds in raw_sets:
                formatted_ds = ds.map(
                    format_conversation,
                    remove_columns=ds.column_names,
                    desc="Formatting conversations",
                )
                filtered_ds = formatted_ds.filter(lambda x: bool(x["text"].strip()))
                if len(filtered_ds) > 0:
                    train_sets.append(filtered_ds)
                else:
                    logger.warning("A dataset had no valid examples after formatting")
            if not train_sets:
                raise RuntimeError("No valid training examples found after formatting/filtering")

            merged: Dataset = datasets.concatenate_datasets(train_sets).shuffle(seed=args.seed)

            def tokenize_function(batch):
                tokenized = tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=block_size,
                    padding=False,
                )
                tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized

            # num_proc: Windows can't use multi-proc here reliably
            num_proc = 1 if os.name == "nt" else min(4, os.cpu_count() or 1)
            tokenized = merged.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                num_proc=num_proc,
                desc="Tokenizing",
            )

            # Pack sequences to full blocks for efficiency
            def group_texts(examples):
                concatenated = list(chain.from_iterable(examples["input_ids"]))
                total_len = (len(concatenated) // block_size) * block_size
                input_ids = [
                    concatenated[i : i + block_size]
                    for i in range(0, total_len, block_size)
                ]
                labels = [ids.copy() for ids in input_ids]
                return {"input_ids": input_ids, "labels": labels}

            train_data = tokenized.map(
                group_texts,
                batched=True,
                desc="Packing to block_size",
            )
            eval_data = None

        # Build model
        model = build_model(tokenizer, block_size)
        n_params = count_trainable_parameters(model)
        logger.info(f"Model has {n_params/1e6:.1f}M trainable parameters")

        # precision selection
        use_bf16 = _gpu_supports_bf16()
        training_args = TrainingArguments(
            output_dir=str(out_dir),
            overwrite_output_dir=(args.mode == "train"),
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            save_steps=500,
            logging_steps=50,
            evaluation_strategy="no",
            bf16=use_bf16,
            fp16=(torch.cuda.is_available() and not use_bf16),
            gradient_checkpointing=True,
            report_to="none",
            save_total_limit=3,
            dataloader_drop_last=True,
            dataloader_num_workers=0 if os.name == "nt" else 2,
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

        logger.info("Starting training...")
        try:
            if args.mode == "resume":
                trainer.train(resume_from_checkpoint=True)
            else:
                trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("Out of memory. Reduce batch size/context, or increase grad_accum.")
                raise
            else:
                raise

        logger.info("Training completed. Saving final model...")
        save_path = out_dir / "final_model"
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)

    elif args.mode == "inference":
        # reload tokenizer; ensure max length is respected
        model_path = Path(args.inference_model_path or Path(args.output_dir) / "final_model")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        tokenizer.padding_side = "left"
        tokenizer.model_max_length = args.context_length

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if _gpu_supports_bf16() else (torch.float16 if torch.cuda.is_available() else torch.float32)

        model = GPT2LMHeadModel.from_pretrained(model_path, torch_dtype=dtype)
        model.to(device)
        model.eval()

        block_size = args.context_length
        input_max = max(16, block_size - args.max_new_tokens)

        print("Chatbot ready - type 'quit' to exit.")
        history = ""

        while True:
            try:
                prompt = input(f"{SPECIAL_TOKENS['user']} ")
                if prompt.lower().strip() in {"quit", "exit"}:
                    break

                current_input = f"{SPECIAL_TOKENS['user']} {prompt}\n{SPECIAL_TOKENS['assistant']}"
                full_input = history + current_input

                inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=input_max)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                print(f"{SPECIAL_TOKENS['assistant']} {response}\n")

                history += current_input + " " + response + f"\n{SPECIAL_TOKENS['end']}\n"
                if len(tokenizer.encode(history)) > block_size // 2:
                    # keep last quarter of tokens to avoid unbounded growth
                    enc = tokenizer.encode(history)
                    history = tokenizer.decode(enc[-block_size // 4 :])

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"[Error] {e}")
                continue

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
