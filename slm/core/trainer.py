"""Enhanced training module with production-grade features."""

import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from loguru import logger
from tqdm import tqdm

from slm.exceptions import TrainingError, CheckpointError
from slm.config import TrainingConfig, ModelConfig, OptimizerType, SchedulerType
from slm.core.models import BaseModel, CharTransformer
from slm.core.data import Vocabulary
from slm.utils import get_device, set_seed, format_duration, get_memory_usage


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum improvement threshold.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.early_stop = False

        logger.debug(f"EarlyStopping initialized: patience={patience}, mode={mode}")

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation score.

        Returns:
            True if training should stop.
        """
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            logger.debug(
                f"Early stopping: improvement detected, best={self.best_score:.6f}"
            )
        else:
            self.counter += 1
            logger.debug(
                f"Early stopping: no improvement, counter={self.counter}/{self.patience}"
            )

        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(
                f"Early stopping triggered after {self.counter} epochs without improvement"
            )

        return self.early_stop


class MetricsTracker:
    """Track and compute training metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        self.memory_usage = []

    def update(
        self,
        loss: float,
        accuracy: Optional[float] = None,
        lr: Optional[float] = None,
        epoch_time: Optional[float] = None,
        memory_info: Optional[Dict[str, float]] = None,
    ):
        """Update metrics."""
        self.losses.append(loss)
        if accuracy is not None:
            self.accuracies.append(accuracy)
        if lr is not None:
            self.learning_rates.append(lr)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
        if memory_info is not None:
            self.memory_usage.append(memory_info)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_epochs": len(self.losses),
            "final_loss": self.losses[-1] if self.losses else None,
            "best_loss": min(self.losses) if self.losses else None,
            "avg_epoch_time": sum(self.epoch_times) / len(self.epoch_times)
            if self.epoch_times
            else None,
            "total_time": sum(self.epoch_times) if self.epoch_times else None,
        }


class CheckpointManager:
    """Manage model checkpoints with validation and recovery."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_every: int = 1,
        max_checkpoints: Optional[int] = None,
        save_best_only: bool = False,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints.
            save_every: Save checkpoint every N epochs.
            max_checkpoints: Maximum number of checkpoints to keep.
            save_best_only: Only save checkpoints with improved validation loss.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_loss = float("inf")
        self.saved_checkpoints = []

        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        model: BaseModel,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        loss: float,
        vocab: Vocabulary,
        config: TrainingConfig,
        model_config: ModelConfig,
        metrics: Dict[str, Any],
        force_save: bool = False,
    ) -> Optional[Path]:
        """Save model checkpoint.

        Args:
            model: Model to save.
            optimizer: Optimizer state.
            scheduler: Learning rate scheduler.
            epoch: Current epoch.
            loss: Current loss.
            vocab: Vocabulary.
            config: Training configuration.
            model_config: Model configuration.
            metrics: Training metrics.
            force_save: Force save regardless of conditions.

        Returns:
            Path to saved checkpoint or None if not saved.
        """
        # Check if we should save
        should_save = (
            force_save
            or (epoch % self.save_every == 0)
            or (self.save_best_only and loss < self.best_loss)
        )

        if not should_save:
            return None

        try:
            # Update best loss
            if loss < self.best_loss:
                self.best_loss = loss
                is_best = True
            else:
                is_best = False

            # Create checkpoint filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}_{timestamp}.pth"
            checkpoint_path = self.checkpoint_dir / checkpoint_name

            # Prepare checkpoint data
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "loss": loss,
                "best_loss": self.best_loss,
                "is_best": is_best,
                "vocab": vocab.chars,
                "config": config.dict(),
                "model_config": model_config.dict(),
                "metrics": metrics,
                "model_info": model.get_model_info(),
                "timestamp": timestamp,
            }

            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            self.saved_checkpoints.append(checkpoint_path)

            # Create best model symlink
            if is_best:
                best_path = self.checkpoint_dir / "best_model.pth"
                if best_path.exists() or best_path.is_symlink():
                    best_path.unlink()
                try:
                    best_path.symlink_to(checkpoint_path.name)
                except OSError:
                    # Fallback for systems without symlink support
                    import shutil

                    shutil.copy2(checkpoint_path, best_path)

            # Clean up old checkpoints
            self._cleanup_checkpoints()

            logger.info(f"Checkpoint saved: {checkpoint_path} (best: {is_best})")
            return checkpoint_path

        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints if exceeding limit."""
        if self.max_checkpoints is None:
            return

        # Sort by creation time
        self.saved_checkpoints.sort(key=lambda p: p.stat().st_ctime)

        # Remove oldest checkpoints
        while len(self.saved_checkpoints) > self.max_checkpoints:
            old_checkpoint = self.saved_checkpoints.pop(0)
            try:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load checkpoint with validation.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Checkpoint data dictionary.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise CheckpointError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Validate checkpoint structure
            required_keys = ["model_state_dict", "epoch", "vocab"]
            for key in required_keys:
                if key not in checkpoint:
                    raise CheckpointError(f"Missing required key in checkpoint: {key}")

            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint

        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint {checkpoint_path}: {e}")


class Trainer:
    """Enhanced trainer with comprehensive features."""

    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        model_config: ModelConfig,
        vocab: Vocabulary,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            config: Training configuration.
            model_config: Model configuration.
            vocab: Vocabulary.
            device: Training device.
        """
        self.model = model
        self.config = config
        self.model_config = model_config
        self.vocab = vocab
        self.device = device or get_device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Initialize components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss()

        # Initialize utilities
        self.early_stopping = None
        if config.early_stopping:
            self.early_stopping = EarlyStopping(config.patience, config.min_delta)

        self.metrics_tracker = MetricsTracker()
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir
            if hasattr(config, "checkpoint_dir")
            else "checkpoints",
            config.save_every,
            save_best_only=config.save_best_only,
        )

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {self.model.get_num_parameters()[0]:,}")

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on configuration."""
        params = self.model.parameters()

        if self.config.optimizer == OptimizerType.ADAM:
            optimizer = torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == OptimizerType.SGD:
            optimizer = torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == OptimizerType.RMSPROP:
            optimizer = torch.optim.RMSprop(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise TrainingError(f"Unsupported optimizer: {self.config.optimizer}")

        logger.info(f"Optimizer created: {self.config.optimizer}")
        return optimizer

    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        if self.config.scheduler == SchedulerType.NONE:
            return None

        if self.config.scheduler == SchedulerType.STEP:
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma,
            )
        elif self.config.scheduler == SchedulerType.COSINE:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )
        elif self.config.scheduler == SchedulerType.EXPONENTIAL:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.gamma,
            )
        elif self.config.scheduler == SchedulerType.REDUCE_ON_PLATEAU:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.gamma,
                patience=self.config.patience // 2,
            )
        else:
            raise TrainingError(f"Unsupported scheduler: {self.config.scheduler}")

        logger.info(f"Scheduler created: {self.config.scheduler}")
        return scheduler

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            try:
                if isinstance(self.model, CharTransformer):
                    logits, _ = self.model(input_ids)
                else:  # CharRNN
                    logits, _ = self.model(input_ids)

                # Compute loss
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1)
                )

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                # Optimizer step
                self.optimizer.step()

                # Update metrics
                batch_loss = loss.item()
                total_loss += batch_loss
                total_tokens += target_ids.numel()

                # Compute accuracy
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    correct_predictions += (predictions == target_ids).sum().item()

                # Update progress bar
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    {"loss": f"{batch_loss:.4f}", "lr": f"{current_lr:.6f}"}
                )

                self.global_step += 1

            except Exception as e:
                logger.error(f"Error in training step {batch_idx}: {e}")
                raise TrainingError(f"Training failed at step {batch_idx}: {e}")

        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_tokens

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0

        with torch.no_grad():
            for input_ids, target_ids in tqdm(val_loader, desc="Validation"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                try:
                    if isinstance(self.model, CharTransformer):
                        logits, _ = self.model(input_ids)
                    else:  # CharRNN
                        logits, _ = self.model(input_ids)

                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)), target_ids.view(-1)
                    )

                    total_loss += loss.item()
                    total_tokens += target_ids.numel()

                    predictions = logits.argmax(dim=-1)
                    correct_predictions += (predictions == target_ids).sum().item()

                except Exception as e:
                    logger.error(f"Error in validation: {e}")
                    raise TrainingError(f"Validation failed: {e}")

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_tokens

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Main training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            resume_from: Path to checkpoint to resume from.

        Returns:
            Training results dictionary.
        """
        # Set random seed
        if hasattr(self.config, "seed") and self.config.seed is not None:
            set_seed(self.config.seed, getattr(self.config, "deterministic", False))

        # Resume from checkpoint if specified
        if resume_from:
            self._resume_from_checkpoint(resume_from)

        logger.info(f"Starting training for {self.config.epochs} epochs")
        start_time = time.time()

        try:
            for epoch in range(self.epoch + 1, self.config.epochs + 1):
                self.epoch = epoch
                epoch_start_time = time.time()

                # Training
                train_metrics = self.train_epoch(train_loader)

                # Validation
                val_metrics = {}
                if val_loader:
                    val_metrics = self.evaluate(val_loader)

                # Learning rate scheduling
                if self.scheduler:
                    if self.config.scheduler == SchedulerType.REDUCE_ON_PLATEAU:
                        self.scheduler.step(
                            val_metrics.get("loss", train_metrics["loss"])
                        )
                    else:
                        self.scheduler.step()

                # Update metrics
                epoch_time = time.time() - epoch_start_time
                memory_info = get_memory_usage()

                self.metrics_tracker.update(
                    loss=train_metrics["loss"],
                    accuracy=train_metrics.get("accuracy"),
                    lr=train_metrics["lr"],
                    epoch_time=epoch_time,
                    memory_info=memory_info,
                )

                # Logging
                log_msg = f"Epoch {epoch}/{self.config.epochs}"
                log_msg += f" | Train Loss: {train_metrics['loss']:.4f}"
                if "accuracy" in train_metrics:
                    log_msg += f" | Train Acc: {train_metrics['accuracy']:.4f}"
                if val_metrics:
                    log_msg += f" | Val Loss: {val_metrics['loss']:.4f}"
                    if "accuracy" in val_metrics:
                        log_msg += f" | Val Acc: {val_metrics['accuracy']:.4f}"
                log_msg += f" | LR: {train_metrics['lr']:.6f}"
                log_msg += f" | Time: {format_duration(epoch_time)}"

                logger.info(log_msg)

                # Save checkpoint
                current_loss = val_metrics.get("loss", train_metrics["loss"])
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    current_loss,
                    self.vocab,
                    self.config,
                    self.model_config,
                    self.metrics_tracker.get_summary(),
                )

                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(current_loss):
                        logger.info("Early stopping triggered")
                        break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise TrainingError(f"Training failed: {e}")

        # Final results
        total_time = time.time() - start_time
        results = {
            "total_epochs": self.epoch,
            "total_time": total_time,
            "final_train_loss": train_metrics["loss"],
            "best_val_loss": self.best_val_loss if val_loader else None,
            "metrics": self.metrics_tracker.get_summary(),
        }

        logger.info(f"Training completed in {format_duration(total_time)}")
        return results

    def _resume_from_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Resume training from checkpoint."""
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            if checkpoint["scheduler_state_dict"] is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training state
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_loss", float("inf"))

        logger.info(f"Resumed training from epoch {self.epoch}")

    def save_model(self, save_path: Union[str, Path]):
        """Save final trained model."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model using the BaseModel method
        self.model.save_pretrained(str(save_path.parent))

        # Also save a simple checkpoint for compatibility
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "vocab": self.vocab.chars,
            "config": self.config.dict(),
            "model_config": self.model_config.dict(),
            "model_info": self.model.get_model_info(),
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
