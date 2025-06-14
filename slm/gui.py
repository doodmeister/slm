"""Enhanced GUI for SLM with production-grade features."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from typing import Dict, Any
from pathlib import Path
import json
import time

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from loguru import logger

from slm.config import Config, ModelType, RNNType, SamplingMethod, create_default_config
from slm.core.data import prepare_data, create_data_loaders
from slm.core.models import create_model
from slm.core.trainer import Trainer
from slm.core.generator import load_generator_from_checkpoint
from slm.utils import get_device, validate_file_path, validate_text_input
from slm.logging_config import setup_logging


class ProgressDialog:
    """Modal progress dialog for long-running operations."""

    def __init__(self, parent, title: str = "Progress", cancelable: bool = False):
        self.parent = parent
        self.cancelable = cancelable
        self.cancelled = False

        # Create modal dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (150 // 2)
        self.dialog.geometry(f"400x150+{x}+{y}")

        # Progress widgets
        self.label = ttk.Label(self.dialog, text="Processing...")
        self.label.pack(pady=10)

        self.progress = ttk.Progressbar(self.dialog, mode="indeterminate")
        self.progress.pack(pady=10, padx=20, fill="x")

        self.details_label = ttk.Label(self.dialog, text="")
        self.details_label.pack(pady=5)

        if cancelable:
            self.cancel_button = ttk.Button(
                self.dialog, text="Cancel", command=self.cancel
            )
            self.cancel_button.pack(pady=10)

        self.progress.start()

    def update(self, message: str, details: str = ""):
        """Update progress dialog."""
        self.label.config(text=message)
        self.details_label.config(text=details)
        self.dialog.update()

    def cancel(self):
        """Cancel operation."""
        self.cancelled = True

    def close(self):
        """Close progress dialog."""
        self.progress.stop()
        self.dialog.grab_release()
        self.dialog.destroy()


class ConfigurationFrame(ttk.LabelFrame):
    """Frame for model and training configuration."""

    def __init__(self, parent, config: Config):
        super().__init__(parent, text="Configuration")
        self.config = config
        self.create_widgets()

    def create_widgets(self):
        """Create configuration widgets."""
        # Model type selection
        model_frame = ttk.LabelFrame(self, text="Model Configuration")
        model_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(model_frame, text="Model Type:").grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.model_type = ttk.Combobox(
            model_frame, values=["rnn", "transformer"], state="readonly"
        )
        self.model_type.set(self.config.model.model_type.value)
        self.model_type.bind("<<ComboboxSelected>>", self.on_model_type_change)
        self.model_type.grid(row=0, column=1, padx=5, pady=2)

        # RNN specific
        ttk.Label(model_frame, text="RNN Type:").grid(
            row=0, column=2, sticky="w", padx=5
        )
        self.rnn_type = ttk.Combobox(
            model_frame, values=["lstm", "gru", "rnn"], state="readonly"
        )
        self.rnn_type.set(self.config.model.rnn_type.value)
        self.rnn_type.grid(row=0, column=3, padx=5, pady=2)

        # Model parameters
        params_frame = ttk.LabelFrame(self, text="Model Parameters")
        params_frame.pack(fill="x", padx=5, pady=5)

        # Create parameter controls
        self.param_vars = {}
        param_configs = [
            ("hidden_dim", "Hidden Dim:", self.config.model.hidden_dim, 64, 1024),
            (
                "embedding_dim",
                "Embedding Dim:",
                self.config.model.embedding_dim,
                32,
                512,
            ),
            ("num_layers", "Layers:", self.config.model.num_layers, 1, 12),
            ("dropout", "Dropout:", self.config.model.dropout, 0.0, 0.8),
        ]

        for i, (name, label, default, min_val, max_val) in enumerate(param_configs):
            row, col = divmod(i, 2)
            ttk.Label(params_frame, text=label).grid(
                row=row, column=col * 2, sticky="w", padx=5, pady=2
            )

            if name == "dropout":
                var = tk.DoubleVar(value=default)
                widget = tk.Spinbox(
                    params_frame,
                    from_=min_val,
                    to=max_val,
                    increment=0.1,
                    width=10,
                    textvariable=var,
                )
            else:
                var = tk.IntVar(value=default)
                widget = tk.Spinbox(
                    params_frame, from_=min_val, to=max_val, width=10, textvariable=var
                )

            widget.grid(row=row, column=col * 2 + 1, padx=5, pady=2)
            self.param_vars[name] = var

        # Training parameters
        train_frame = ttk.LabelFrame(self, text="Training Parameters")
        train_frame.pack(fill="x", padx=5, pady=5)

        train_configs = [
            ("epochs", "Epochs:", self.config.training.epochs, 1, 1000),
            ("batch_size", "Batch Size:", self.config.training.batch_size, 1, 512),
            ("seq_length", "Seq Length:", self.config.training.seq_length, 10, 1000),
            (
                "learning_rate",
                "Learning Rate:",
                self.config.training.learning_rate,
                0.0001,
                1.0,
            ),
        ]

        for i, (name, label, default, min_val, max_val) in enumerate(train_configs):
            row, col = divmod(i, 2)
            ttk.Label(train_frame, text=label).grid(
                row=row, column=col * 2, sticky="w", padx=5, pady=2
            )

            if name == "learning_rate":
                var = tk.DoubleVar(value=default)
                widget = tk.Spinbox(
                    train_frame,
                    from_=min_val,
                    to=max_val,
                    increment=0.001,
                    width=10,
                    textvariable=var,
                    format="%.4f",
                )
            else:
                var = tk.IntVar(value=default)
                widget = tk.Spinbox(
                    train_frame, from_=min_val, to=max_val, width=10, textvariable=var
                )

            widget.grid(row=row, column=col * 2 + 1, padx=5, pady=2)
            self.param_vars[name] = var

    def on_model_type_change(self, event=None):
        """Handle model type change."""
        model_type = self.model_type.get()
        # Enable/disable RNN type based on model type
        if model_type == "transformer":
            self.rnn_type.config(state="disabled")
        else:
            self.rnn_type.config(state="readonly")

    def get_config(self) -> Config:
        """Get configuration from widgets."""
        config = create_default_config()

        # Model configuration
        config.model.model_type = ModelType(self.model_type.get())
        config.model.rnn_type = RNNType(self.rnn_type.get())

        # Update parameters
        for name, var in self.param_vars.items():
            if hasattr(config.model, name):
                setattr(config.model, name, var.get())
            elif hasattr(config.training, name):
                setattr(config.training, name, var.get())

        return config


class TrainingTab(ttk.Frame):
    """Training tab with enhanced features."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.training_thread = None
        self.trainer = None
        self.create_widgets()

    def create_widgets(self):
        """Create training widgets."""
        # File selection
        file_frame = ttk.LabelFrame(self, text="Training Data")
        file_frame.pack(fill="x", padx=10, pady=5)

        self.file_path = tk.StringVar()
        ttk.Button(file_frame, text="Select File", command=self.select_file).pack(
            side="left", padx=5, pady=5
        )

        ttk.Label(file_frame, textvariable=self.file_path).pack(
            side="left", padx=5, pady=5, fill="x", expand=True
        )

        # Configuration frame
        self.config_frame = ConfigurationFrame(self, self.app.config)
        self.config_frame.pack(fill="x", padx=10, pady=5)

        # Control buttons
        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.train_button = ttk.Button(
            control_frame, text="Start Training", command=self.start_training
        )
        self.train_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Training",
            command=self.stop_training,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=5)

        ttk.Button(control_frame, text="Save Config", command=self.save_config).pack(
            side="left", padx=5
        )

        ttk.Button(control_frame, text="Load Config", command=self.load_config).pack(
            side="left", padx=5
        )

        # Status and progress
        status_frame = ttk.LabelFrame(self, text="Training Status")
        status_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.status_text = scrolledtext.ScrolledText(
            status_frame, height=10, state="disabled"
        )
        self.status_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Progress frame
        progress_frame = ttk.Frame(status_frame)
        progress_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(progress_frame, text="Progress:").pack(side="left")
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(
            side="left", padx=10
        )
        # Training metrics plot
        if HAS_MATPLOTLIB:
            self.create_plot_frame(status_frame)

    def create_plot_frame(self, parent):
        """Create matplotlib plot for training metrics."""
        plot_frame = ttk.LabelFrame(parent, text="Training Metrics")
        plot_frame.pack(fill="both", expand=True, padx=5, pady=5)

        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(8, 4), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title("Training Loss")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss")

            self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            # Fallback when matplotlib is not available
            fallback_label = ttk.Label(plot_frame, text="Matplotlib not available")
            fallback_label.pack(expand=True)

        self.losses = []

    def select_file(self):
        """Select training data file."""
        file_path = filedialog.askopenfilename(
            title="Select Training Data",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if file_path:
            try:
                validate_file_path(file_path, must_exist=True)
                self.file_path.set(file_path)
                self.log_message(f"Selected training data: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Invalid file: {e}")

    def save_config(self):
        """Save current configuration."""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[
                ("YAML files", "*.yaml"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                config = self.config_frame.get_config()
                if file_path.endswith(".json"):
                    with open(file_path, "w") as f:
                        json.dump(config.dict(), f, indent=2)
                else:
                    from slm.config import save_config

                    save_config(config, file_path)

                self.log_message(f"Configuration saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self):
        """Load configuration from file."""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[
                ("YAML files", "*.yaml"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                from slm.config import load_config

                config = load_config(file_path)

                # Update UI with loaded config
                self.config_frame.model_type.set(config.model.model_type.value)
                self.config_frame.rnn_type.set(config.model.rnn_type.value)

                for name, var in self.config_frame.param_vars.items():
                    if hasattr(config.model, name):
                        var.set(getattr(config.model, name))
                    elif hasattr(config.training, name):
                        var.set(getattr(config.training, name))

                self.log_message(f"Configuration loaded from: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def start_training(self):
        """Start training in a separate thread."""
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a training data file")
            return

        try:
            # Validate configuration
            config = self.config_frame.get_config()
            config.data_path = Path(self.file_path.get())

            # Start training thread
            self.training_thread = threading.Thread(
                target=self.train_model, args=(config,), daemon=True
            )
            self.training_thread.start()

            # Update UI state
            self.train_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.progress_var.set("Training in progress...")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")

    def stop_training(self):
        """Stop training."""
        if self.trainer and hasattr(self.trainer, 'stop_requested'):
            self.trainer.stop_requested = True
        elif self.training_thread and self.training_thread.is_alive():
            # If trainer doesn't have stop_requested, we can only update UI
            self.log_message("Training stop requested - will complete current epoch")

        self.train_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_var.set("Training stop requested")
        self.log_message("Training stop requested by user")

    def train_model(self, config: Config):
        """Train model (runs in separate thread)."""
        try:
            self.log_message("Starting training preparation...")

            # Validate data path
            if config.data_path is None:
                raise ValueError("No data path specified in configuration")

            # Prepare data
            text, vocab = prepare_data(config.data_path, config.training.seq_length)

            train_loader, val_loader, data_info = create_data_loaders(
                text,
                vocab,
                config.training.seq_length,
                config.training.batch_size,
                config.training.validation_split,
            )

            self.log_message(
                f"Data prepared: {data_info['total_chars']} characters, "
                f"{data_info['vocab_size']} vocabulary size"
            )

            # Create model
            config.model.vocab_size = len(vocab)
            model = create_model(config.model)

            self.log_message(
                f"Model created: {model.get_num_parameters()[0]:,} parameters"
            )

            # Create trainer
            device = get_device(config.training.device)
            self.trainer = Trainer(model, config.training, config.model, vocab, device)

            # Custom progress callback
            def epoch_callback(epoch, loss, val_loss=None):
                self.after_training_epoch(epoch, loss, val_loss)

            # Start training
            self.log_message("Starting training...")
            results = self.trainer.train(train_loader, val_loader)

            self.log_message("Training completed successfully!")
            self.log_message(
                f"Final loss: {results.get('final_train_loss', 'N/A'):.6f}"
            )

            # Update UI state
            self.app.root.after(0, self.training_completed, True)

        except Exception as e:
            logger.exception("Training failed")
            self.log_message(f"Training failed: {e}")
            self.app.root.after(0, self.training_completed, False)

    def after_training_epoch(self, epoch, loss, val_loss=None):
        """Called after each training epoch."""

        def update_ui():
            self.progress_var.set(f"Epoch {epoch} - Loss: {loss:.6f}")

            if HAS_MATPLOTLIB:
                self.losses.append(loss)
                self.update_plot()

        self.app.root.after(0, update_ui)

    def update_plot(self):
        """Update training loss plot."""
        if not HAS_MATPLOTLIB or not self.losses:
            return

        self.ax.clear()
        self.ax.plot(self.losses, "b-", label="Training Loss")
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def training_completed(self, success: bool):
        """Called when training is completed."""
        self.train_button.config(state="normal")
        self.stop_button.config(state="disabled")

        if success:
            self.progress_var.set("Training completed")
            messagebox.showinfo("Success", "Training completed successfully!")
        else:
            self.progress_var.set("Training failed")

    def log_message(self, message: str):
        """Add message to status log."""

        def update_log():
            self.status_text.config(state="normal")
            timestamp = time.strftime("%H:%M:%S")
            self.status_text.insert("end", f"[{timestamp}] {message}\n")
            self.status_text.see("end")
            self.status_text.config(state="disabled")

        if threading.current_thread() != threading.main_thread():
            self.app.root.after(0, update_log)
        else:
            update_log()


class GenerationTab(ttk.Frame):
    """Text generation tab with enhanced features."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.generator = None
        self.create_widgets()

    def create_widgets(self):
        """Create generation widgets."""
        # Model loading
        model_frame = ttk.LabelFrame(self, text="Model")
        model_frame.pack(fill="x", padx=10, pady=5)

        self.model_path = tk.StringVar()
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(
            side="left", padx=5, pady=5
        )

        ttk.Label(model_frame, textvariable=self.model_path).pack(
            side="left", padx=5, pady=5, fill="x", expand=True
        )

        # Generation parameters
        params_frame = ttk.LabelFrame(self, text="Generation Parameters")
        params_frame.pack(fill="x", padx=10, pady=5)

        # Parameter controls
        param_configs = [
            ("length", "Length:", 200, 1, 10000),
            ("temperature", "Temperature:", 1.0, 0.1, 2.0),
            ("top_k", "Top-k:", 0, 0, 100),
            ("top_p", "Top-p:", 0.9, 0.1, 1.0),
        ]

        self.gen_params = {}
        for i, (name, label, default, min_val, max_val) in enumerate(param_configs):
            row, col = divmod(i, 2)
            ttk.Label(params_frame, text=label).grid(
                row=row, column=col * 2, sticky="w", padx=5, pady=2
            )

            if name in ["temperature", "top_p"]:
                var = tk.DoubleVar(value=default)
                widget = tk.Spinbox(
                    params_frame,
                    from_=min_val,
                    to=max_val,
                    increment=0.1,
                    width=10,
                    textvariable=var,
                    format="%.1f",
                )
            else:
                var = tk.IntVar(value=default)
                widget = tk.Spinbox(
                    params_frame, from_=min_val, to=max_val, width=10, textvariable=var
                )

            widget.grid(row=row, column=col * 2 + 1, padx=5, pady=2)
            self.gen_params[name] = var

        # Sampling method
        ttk.Label(params_frame, text="Sampling:").grid(
            row=2, column=0, sticky="w", padx=5, pady=2
        )
        self.sampling_method = ttk.Combobox(
            params_frame,
            values=["greedy", "temperature", "top_k", "top_p"],
            state="readonly",
        )
        self.sampling_method.set("temperature")
        self.sampling_method.grid(row=2, column=1, padx=5, pady=2)

        # Prompt input
        prompt_frame = ttk.LabelFrame(self, text="Prompt")
        prompt_frame.pack(fill="x", padx=10, pady=5)

        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=4)
        self.prompt_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Control buttons
        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.generate_button = ttk.Button(
            control_frame, text="Generate", command=self.generate_text, state="disabled"
        )
        self.generate_button.pack(side="left", padx=5)

        ttk.Button(control_frame, text="Clear", command=self.clear_output).pack(
            side="left", padx=5
        )

        ttk.Button(control_frame, text="Save Output", command=self.save_output).pack(
            side="left", padx=5
        )

        # Output
        output_frame = ttk.LabelFrame(self, text="Generated Text")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.output_text = scrolledtext.ScrolledText(output_frame, state="disabled")
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)
        # Status
        self.status_var = tk.StringVar(value="No model loaded")
        status_label = ttk.Label(self, textvariable=self.status_var)
        status_label.pack(pady=5)

    def load_model(self):
        """Load a trained model."""
        file_path = filedialog.askopenfilename(
            title="Load Model Checkpoint",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")],
        )

        if file_path:
            try:
                progress = ProgressDialog(self, "Loading Model...")
                progress.update("Loading model checkpoint...")

                def load_in_thread():
                    try:
                        from slm.config import GenerationConfig

                        gen_config = GenerationConfig()
                        self.generator = load_generator_from_checkpoint(
                            file_path, gen_config
                        )
                        self.app.root.after(
                            0, lambda: self.model_loaded(file_path, progress)
                        )
                    except Exception as e:
                        error_msg = str(e)
                        self.app.root.after(
                            0, lambda: self.model_load_failed(error_msg, progress)
                        )

                threading.Thread(target=load_in_thread, daemon=True).start()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")

    def model_loaded(self, file_path: str, progress: ProgressDialog):
        """Called when model is successfully loaded."""
        progress.close()
        self.model_path.set(file_path)
        self.generate_button.config(state="normal")
        self.status_var.set("Model loaded successfully")
        messagebox.showinfo("Success", "Model loaded successfully!")

    def model_load_failed(self, error: str, progress: ProgressDialog):
        """Called when model loading fails."""
        progress.close()
        self.status_var.set("Failed to load model")
        messagebox.showerror("Error", f"Failed to load model: {error}")

    def generate_text(self):
        """Generate text using the loaded model."""
        if not self.generator:
            messagebox.showerror("Error", "No model loaded")
            return

        try:
            # Get parameters
            prompt = self.prompt_text.get("1.0", "end-1c")

            # Validate prompt
            try:
                validate_text_input(prompt, min_length=0, max_length=10000)
            except Exception as e:
                messagebox.showerror("Error", f"Invalid prompt: {e}")
                return

            # Additional safety check
            if self.generator is None:
                messagebox.showerror("Error", "Model generator is not available")
                self.generate_button.config(state="disabled")
                self.status_var.set("No model loaded")
                return

            # Update generation config
            self.generator.config.length = self.gen_params["length"].get()
            self.generator.config.temperature = self.gen_params["temperature"].get()
            self.generator.config.top_k = self.gen_params["top_k"].get()
            self.generator.config.top_p = self.gen_params["top_p"].get()
            self.generator.config.sampling_method = SamplingMethod(
                self.sampling_method.get()
            )
            # Recreate sampling strategy with new config
            self.generator.sampling_strategy = (
                self.generator._create_sampling_strategy()
            )
            # Start generation in thread
            self.generate_button.config(state="disabled")
            self.status_var.set("Generating text...")

            def generate_in_thread():
                try:
                    # Final check before generation
                    if self.generator is None:
                        raise RuntimeError("Generator became unavailable during execution")
                    result = self.generator.generate(prompt)
                    self.app.root.after(0, lambda: self.generation_completed(result))
                except Exception as e:
                    error_msg = str(e)
                    self.app.root.after(0, lambda: self.generation_failed(error_msg))

            threading.Thread(target=generate_in_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Generation failed: {e}")
            self.generate_button.config(state="normal")

    def generation_completed(self, result: Dict[str, Any]):
        """Called when generation is completed."""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", result["generated_text"])
        self.output_text.config(state="disabled")

        self.generate_button.config(state="normal")
        self.status_var.set(
            f"Generated {result['new_tokens']} tokens in "
            f"{result['generation_time']:.2f}s "
            f"({result['tokens_per_second']:.1f} tokens/s)"
        )

    def generation_failed(self, error: str):
        """Called when generation fails."""
        self.generate_button.config(state="normal")
        self.status_var.set("Generation failed")
        messagebox.showerror("Error", f"Generation failed: {error}")

    def clear_output(self):
        """Clear the output text."""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.config(state="disabled")

    def save_output(self):
        """Save generated text to file."""
        text = self.output_text.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showwarning("Warning", "No text to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Generated Text",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                messagebox.showinfo("Success", f"Text saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")


class SLMApplication:
    """Main SLM GUI application."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simple Language Models - Production GUI")
        self.root.geometry("1000x800")
        # Setup logging
        try:
            setup_logging(level="INFO", log_to_file=True, log_dir="logs")
        except Exception:
            pass  # Continue without logging if setup fails

        # Initialize configuration
        self.config = create_default_config()

        # Create GUI
        self.create_widgets()

        # Setup menu
        self.create_menu()

        logger.info("SLM GUI application initialized")

    def create_widgets(self):
        """Create main application widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # Training tab
        self.training_tab = TrainingTab(self.notebook, self)
        self.notebook.add(self.training_tab, text="Training")

        # Generation tab
        self.generation_tab = GenerationTab(self.notebook, self)
        self.notebook.add(self.generation_tab, text="Generation")

        # Status bar
        self.status_bar = ttk.Label(
            self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Config", command=self.new_config)
        file_menu.add_command(label="Load Config", command=self.load_config)
        file_menu.add_command(label="Save Config", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def new_config(self):
        """Create new default configuration."""
        self.config = create_default_config()
        messagebox.showinfo("Info", "Configuration reset to defaults")

    def load_config(self):
        """Load configuration from file."""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[
                ("YAML files", "*.yaml"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                from slm.config import load_config

                self.config = load_config(file_path)
                messagebox.showinfo(
                    "Success", f"Configuration loaded from: {file_path}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def save_config(self):
        """Save current configuration to file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[
                ("YAML files", "*.yaml"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                from slm.config import save_config

                save_config(self.config, file_path)
                messagebox.showinfo("Success", f"Configuration saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def show_about(self):
        """Show about dialog."""
        about_text = """
Simple Language Models (SLM) v2.0
Production-grade character-level language models

Features:
• RNN and Transformer architectures
• Advanced training with early stopping
• Multiple text generation strategies
• Comprehensive configuration system
• Production-ready error handling

Built with PyTorch and Tkinter
        """
        messagebox.showinfo("About SLM", about_text.strip())

    def run(self):
        """Run the application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.exception("Application error")
            messagebox.showerror("Error", f"Application error: {e}")


def main():
    """Main entry point for GUI application."""
    try:
        app = SLMApplication()
        app.run()
    except Exception as e:
        print(f"Failed to start GUI application: {e}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
