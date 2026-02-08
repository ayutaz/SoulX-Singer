"""Training script for SoulX-Singer.

Usage:
    # Fine-tuning (recommended)
    accelerate launch -m cli.train \
        --data_dir data/japanese_songs \
        --resume_from pretrained_models/SoulX-Singer/model.pt \
        --config soulxsinger/config/soulxsinger.yaml \
        --save_dir checkpoints/ja_finetune

    # Full training (from scratch)
    accelerate launch -m cli.train \
        --data_dir data/all_songs \
        --config soulxsinger/config/soulxsinger.yaml \
        --save_dir checkpoints/full_train

    # Override config values via CLI
    accelerate launch -m cli.train \
        --data_dir data/songs \
        --config soulxsinger/config/soulxsinger.yaml \
        --save_dir checkpoints/run1 \
        --max_steps 10
"""

import os
import math
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed

from soulxsinger.utils.file_utils import load_config
from soulxsinger.models.soulxsinger import SoulXSinger
from soulxsinger.utils.data_processor import DataProcessor
from soulxsinger.utils.dataset import SVSDataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SoulX-Singer")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to training dataset directory")
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Path to validation dataset directory (optional)")
    parser.add_argument("--config", type=str, default="soulxsinger/config/soulxsinger.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--phoneset_path", type=str,
                        default="soulxsinger/utils/phoneme/phone_set.json",
                        help="Path to phoneme set JSON")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume/fine-tune from")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    # CLI overrides for train config
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--freeze_encoder", action="store_true", default=None)
    parser.add_argument("--freeze_transformer", action="store_true", default=None)
    parser.add_argument("--wandb_project", type=str, default="soulx-singer")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model, optimizer, scheduler, step, config, save_path, accelerator):
    """Save checkpoint compatible with inference pipeline."""
    unwrapped = accelerator.unwrap_model(model)
    state = {
        "state_dict": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }
    accelerator.save(state, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="wandb" if not args.no_wandb else None,
        mixed_precision="no",
    )
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
    )

    # Load config
    config = load_config(args.config)
    train_cfg = config.train

    # CLI overrides
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.freeze_encoder is not None:
        train_cfg.freeze_encoder = args.freeze_encoder
    if args.freeze_transformer is not None:
        train_cfg.freeze_transformer = args.freeze_transformer

    # Build model
    model = SoulXSinger(config)
    logger.info(f"Model parameters: {count_params(model, trainable_only=False) / 1e6:.1f}M")

    # Load checkpoint for fine-tuning
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, weights_only=False, map_location="cpu")
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            logger.info(f"Loaded model weights from {args.resume_from}")
        else:
            logger.warning(f"No 'state_dict' in checkpoint {args.resume_from}, skipping weight load")

    # Freeze vocoder (always)
    if train_cfg.freeze_vocoder:
        freeze_params(model.vocoder)
        model.vocoder.eval()
        logger.info("Vocoder frozen")

    # Freeze encoder (for fine-tuning)
    if train_cfg.freeze_encoder:
        freeze_params(model.note_text_encoder)
        freeze_params(model.note_pitch_encoder)
        freeze_params(model.note_type_encoder)
        freeze_params(model.f0_encoder)
        freeze_params(model.preflow)
        logger.info("Encoder frozen")

    # Freeze transformer (optional)
    if train_cfg.freeze_transformer:
        freeze_params(model.cfm_decoder)
        logger.info("Transformer frozen")

    logger.info(f"Trainable parameters: {count_params(model, trainable_only=True) / 1e6:.1f}M")

    # Build dataset and dataloader
    data_processor = DataProcessor(
        hop_size=config.audio.hop_size,
        sample_rate=config.audio.sample_rate,
        phoneset_path=args.phoneset_path,
        device="cpu",  # Move to device later via accelerator
    )

    train_dataset = SVSDataset(
        data_dir=args.data_dir,
        data_processor=data_processor,
        sample_rate=config.audio.sample_rate,
        hop_size=config.audio.hop_size,
    )
    logger.info(f"Training dataset: {len(train_dataset)} samples")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=SVSDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = None
    if args.val_data_dir:
        val_dataset = SVSDataset(
            data_dir=args.val_data_dir,
            data_processor=data_processor,
            sample_rate=config.audio.sample_rate,
            hop_size=config.audio.hop_size,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=SVSDataset.collate_fn,
            pin_memory=True,
        )
        logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=train_cfg.max_steps,
    )

    # Resume optimizer/scheduler state if continuing
    global_step = 0
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, weights_only=False, map_location="cpu")
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("Loaded optimizer state")
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("Loaded scheduler state")
        if "step" in checkpoint:
            global_step = checkpoint["step"]
            logger.info(f"Resuming from step {global_step}")

    # Prepare with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # wandb init
    if not args.no_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config={
                "train": dict(train_cfg),
                "model": dict(config.model),
                "audio": dict(config.audio),
                "resume_from": args.resume_from,
                "data_dir": args.data_dir,
            },
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )

    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    logger.info("Starting training...")
    model.train()
    # Keep vocoder in eval mode
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.vocoder.eval()

    data_iter = iter(train_dataloader)

    while global_step < train_cfg.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            outputs = model(batch)
            loss = outputs["loss"]
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg.grad_clip,
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        # Logging
        if global_step % train_cfg.log_every == 0 and accelerator.is_main_process:
            lr = scheduler.get_last_lr()[0]
            log_dict = {
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/step": global_step,
            }
            if accelerator.sync_gradients and isinstance(grad_norm, (int, float, torch.Tensor)):
                log_dict["train/grad_norm"] = float(grad_norm)
            logger.info(
                f"Step {global_step}/{train_cfg.max_steps} | "
                f"loss={loss.item():.4f} | lr={lr:.2e}"
            )
            if not args.no_wandb:
                accelerator.log(log_dict, step=global_step)

        # Validation
        if (val_dataloader is not None
                and global_step % train_cfg.eval_every == 0
                and accelerator.is_main_process):
            model.eval()
            val_losses = []
            for val_batch in val_dataloader:
                with torch.no_grad():
                    val_out = model(val_batch)
                    val_losses.append(val_out["loss"].item())
            val_loss = sum(val_losses) / len(val_losses)
            logger.info(f"Step {global_step} | val_loss={val_loss:.4f}")
            if not args.no_wandb:
                accelerator.log({"val/loss": val_loss}, step=global_step)
            model.train()
            unwrapped.vocoder.eval()

        # Save checkpoint
        if global_step % train_cfg.save_every == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_step{global_step}.pt")
            save_checkpoint(model, optimizer, scheduler, global_step, config, ckpt_path, accelerator)

    # Final save
    if accelerator.is_main_process:
        ckpt_path = os.path.join(args.save_dir, f"checkpoint_step{global_step}.pt")
        save_checkpoint(model, optimizer, scheduler, global_step, config, ckpt_path, accelerator)

    if not args.no_wandb:
        accelerator.end_training()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
