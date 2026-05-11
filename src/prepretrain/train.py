"""
Training script for pre-pretraining experiments.
Based on the original paper's implementation but self-contained in formal_langauge.
"""
from typing import List
import json
import os

import datasets
import fire
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


class SaveAtStepsCallback(TrainerCallback):
    """Custom callback to save model at specific training steps."""

    def __init__(self, save_steps: List[int], output_dir: str):
        self.save_steps = sorted(save_steps)
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.save_steps:
            checkpoint_dir = f"{self.output_dir}/checkpoint-{state.global_step}"
            kwargs["model"].save_pretrained(checkpoint_dir)
            if "tokenizer" in kwargs:
                kwargs["tokenizer"].save_pretrained(checkpoint_dir)
            print(f"Saved model at step {state.global_step}")


def main(
    data_dir: str = "./data/tokenized/depth9_train",
    model_name: str = "EleutherAI/pythia-160m",
    model_path: str = None,  # Path to load pretrained model from (for C4 stage)
    reinit: bool = False,
    max_seq_length: int = 2048,
    gradient_accumulation_steps: int = 1,
    max_steps: int = 10000,
    bsz: int = 32,
    warmup_steps: int = 500,
    logging_steps: int = 5,
    save_steps: int = 250,
    save_only_model: bool = True,
    output_dir: str = "output",
    seed: int = 3407,
    report_to: str = "wandb",
    lr: float = 1e-3,
    min_lr_rate: float = 0.1,
    weight_decay: float = 0.0,
    packing_mode: str = "pack",
    override_packing: bool = False,
    attn_implementation: str = "flash_attention_2",
    bf16: bool = True,
    use_callback: bool = False,
    eval_data_dir: str = None,
    eval_steps: int = 100,
    eval_samples: int = 10000,
):
    """
    Main training function.

    Args:
        data_dir: Path to tokenized dataset
        model_name: HuggingFace model name
        model_path: Path to load model checkpoint (for C4 training after PPT)
        reinit: Whether to reinitialize model weights
        max_seq_length: Maximum sequence length
        gradient_accumulation_steps: Gradient accumulation steps
        max_steps: Maximum training steps
        bsz: Batch size
        warmup_steps: Warmup steps
        logging_steps: Logging frequency
        save_steps: Checkpoint save frequency
        save_only_model: Only save model weights in checkpoints (no optimizer/scheduler/rng states)
        output_dir: Output directory
        seed: Random seed
        report_to: Reporting backend (wandb/none)
        lr: Learning rate
        min_lr_rate: Minimum learning rate ratio for cosine scheduler
        weight_decay: Weight decay
        packing_mode: Packing strategy: auto/pack/unpack
        override_packing: Legacy flag; if True, force unpacking
        attn_implementation: Attention backend passed to transformers. Use "eager"
            if flash-attn is not installed.
        bf16: Train in bfloat16
        use_callback: Use custom save callback
        eval_data_dir: Evaluation dataset directory
        eval_steps: Evaluation frequency
        eval_samples: Number of eval samples to use
    """
    print(f"Training config: {locals()}")
    set_seed(seed)
    print(f"Seed set: {seed}")

    callback = SaveAtStepsCallback(
        save_steps=list(range(0, 4000, 100)) + list(range(4000, 10000, 1000)),
        output_dir=output_dir,
    )

    # Load training dataset
    print(f"Loading train dataset from: {data_dir}")
    dataset = datasets.load_from_disk(data_dir)
    if "train" in dataset:
        dataset = dataset["train"]
    print(f"Train dataset size: {len(dataset)}")

    # Load eval dataset if specified
    eval_dataset = None
    if eval_data_dir is not None:
        print(f"Loading eval dataset from: {eval_data_dir}")
        eval_dataset = datasets.load_from_disk(eval_data_dir)
        if "train" in eval_dataset:
            eval_dataset = eval_dataset["train"]
        if eval_samples is not None and eval_samples > 0 and len(eval_dataset) > eval_samples:
            eval_dataset = eval_dataset.select(range(eval_samples))
        print(f"Eval dataset size: {len(eval_dataset)}")

    model_kwargs = {"torch_dtype": torch.bfloat16 if bf16 else torch.float32}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    # Load model
    if model_path is not None:
        # Load from checkpoint (for C4 training after PPT)
        print(f"Loading model from checkpoint: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs,
        )
    elif reinit:
        # Random initialization
        print(f"Reinitializing model from config: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        print(f"Loaded config: {config.__class__.__name__}")
        model = AutoModelForCausalLM.from_config(
            config,
            **model_kwargs,
        )
    else:
        # Load pretrained
        print(f"Loading pretrained model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these experiments.")
    print("Moving model to CUDA")
    model.cuda()
    print("Model on CUDA")

    print(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    print("Tokenizer ready")

    # Determine packing strategy
    if packing_mode not in {"auto", "pack", "unpack"}:
        raise ValueError(
            f"Invalid packing_mode='{packing_mode}'. "
            "Expected one of: auto, pack, unpack"
        )

    if packing_mode == "auto":
        packing = "c4" in data_dir.lower()
    elif packing_mode == "pack":
        packing = True
    else:
        packing = False

    if override_packing:
        packing = False

    training_kwargs = {
        "per_device_train_batch_size": bsz,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "logging_steps": logging_steps,
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": eval_steps if eval_dataset is not None else 0,
        "eval_on_start": True if eval_dataset is not None else False,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "output_dir": output_dir,
        "seed": seed,
        "report_to": report_to,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "lr_scheduler_type": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {"min_lr_rate": min_lr_rate},
        "packing": packing,
        "max_length": max_seq_length,
        "bf16": bf16,
    }
    if "save_only_model" in SFTConfig.__dataclass_fields__:
        training_kwargs["save_only_model"] = save_only_model
    elif save_only_model:
        print(
            "Warning: current TRL SFTConfig does not support save_only_model; "
            "optimizer/scheduler states may still be saved."
        )

    training_args = SFTConfig(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
    )
    print("Trainer initialized")
    if use_callback:
        trainer.add_callback(callback)

    trainer.train()

    # Save training history
    history = {"step": [], "train_loss": [], "eval_loss": []}
    last_train_loss = None
    for entry in trainer.state.log_history:
        if "loss" in entry and "eval_loss" not in entry:
            last_train_loss = entry["loss"]
        if "eval_loss" in entry:
            history["step"].append(int(entry.get("step", 0)))
            history["train_loss"].append(last_train_loss)
            history["eval_loss"].append(entry["eval_loss"])

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))
    print(f"Training complete. History saved to {output_dir}/history.json")


if __name__ == "__main__":
    fire.Fire(main)
