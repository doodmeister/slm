"""Command Line Interface for SLM."""

from pathlib import Path
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from slm.config import (
    GenerationConfig,
    ModelType,
    RNNType,
    SamplingMethod,
    load_config,
    save_config,
    create_default_config,
)
from slm.core.data import prepare_data, create_data_loaders
from slm.core.models import create_model
from slm.core.trainer import Trainer
from slm.core.generator import load_generator_from_checkpoint
from slm.utils import get_device
from slm.logging_config import setup_logging
from slm.exceptions import SLMException

console = Console()


def setup_cli_logging(log_level: str = "INFO", log_to_file: bool = True):
    """Setup logging for CLI."""
    try:
        setup_logging(
            level=log_level,
            log_to_file=log_to_file,
            log_dir="logs",
            log_filename="slm_cli.log",
        )
    except Exception as e:
        console.print(f"[red]Warning: Could not setup logging: {e}[/red]")


@click.group()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Configuration file path"
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
@click.option("--no-log-file", is_flag=True, help="Disable file logging")
@click.pass_context
def cli(ctx, config, log_level, no_log_file):
    """Simple Language Models (SLM) - Production-grade character-level language models."""
    # Setup logging
    setup_cli_logging(log_level, not no_log_file)

    # Load configuration
    if config:
        try:
            ctx.obj = load_config(config)
            console.print(f"[green]Configuration loaded from: {config}[/green]")
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            ctx.obj = create_default_config()
    else:
        ctx.obj = create_default_config()


@cli.command()
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Training data file",
)
@click.option("--model-type", type=click.Choice(["rnn", "transformer"]), default="rnn")
@click.option("--rnn-type", type=click.Choice(["lstm", "gru", "rnn"]), default="lstm")
@click.option("--hidden-dim", type=int, default=256, help="Hidden/model dimension")
@click.option("--embedding-dim", type=int, default=128, help="Embedding dimension")
@click.option("--num-layers", type=int, default=2, help="Number of layers")
@click.option("--epochs", type=int, default=10, help="Number of training epochs")
@click.option("--batch-size", type=int, default=64, help="Batch size")
@click.option("--learning-rate", type=float, default=0.002, help="Learning rate")
@click.option("--seq-length", type=int, default=100, help="Sequence length")
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    default="checkpoints",
    help="Checkpoint directory",
)
@click.option("--resume", type=click.Path(exists=True), help="Resume from checkpoint")
@click.option("--seed", type=int, help="Random seed")
@click.option("--device", help="Device to use (cpu, cuda, cuda:0, etc.)")
@click.option("--save-config", type=click.Path(), help="Save configuration to file")
@click.pass_context
def train(ctx, **kwargs):
    """Train a language model."""
    config = ctx.obj

    try:
        # Update configuration with CLI arguments
        if kwargs.get("model_type"):
            config.model.model_type = ModelType(kwargs["model_type"])

        if kwargs.get("rnn_type"):
            config.model.rnn_type = RNNType(kwargs["rnn_type"])

        # Update model config
        model_updates = {
            "hidden_dim": kwargs.get("hidden_dim"),
            "embedding_dim": kwargs.get("embedding_dim"),
            "num_layers": kwargs.get("num_layers"),
        }
        for key, value in model_updates.items():
            if value is not None:
                setattr(config.model, key, value)

        # Update training config
        training_updates = {
            "epochs": kwargs.get("epochs"),
            "batch_size": kwargs.get("batch_size"),
            "learning_rate": kwargs.get("learning_rate"),
            "seq_length": kwargs.get("seq_length"),
            "device": kwargs.get("device"),
        }
        for key, value in training_updates.items():
            if value is not None:
                setattr(config.training, key, value)

        # Set paths
        config.data_path = Path(kwargs["data"])
        config.checkpoint_dir = Path(kwargs["checkpoint_dir"])

        # Set seed
        if kwargs.get("seed"):
            config.seed = kwargs["seed"]

        # Save config if requested
        if kwargs.get("save_config"):
            save_config(config, kwargs["save_config"])
            console.print(
                f"[green]Configuration saved to: {kwargs['save_config']}[/green]"
            )

        # Start training
        with console.status("[bold green]Preparing training data..."):
            text, vocab = prepare_data(
                config.data_path,
                config.training.seq_length,
                cache_dir=config.checkpoint_dir / "cache",
            )

            train_loader, val_loader, data_info = create_data_loaders(
                text,
                vocab,
                config.training.seq_length,
                config.training.batch_size,
                config.training.validation_split,
                num_workers=config.training.num_workers,
                seed=config.seed,
            )

        # Display data info
        table = Table(title="Training Data Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in data_info.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)

        # Create model
        config.model.vocab_size = len(vocab)
        model = create_model(config.model)

        # Display model info
        model_info = model.get_model_info()
        model_table = Table(title="Model Information")
        model_table.add_column("Property", style="cyan")
        model_table.add_column("Value", style="magenta")

        for key, value in model_info.items():
            if key != "config":
                model_table.add_row(key.replace("_", " ").title(), str(value))

        console.print(model_table)

        # Initialize trainer
        device = get_device(config.training.device)
        trainer = Trainer(model, config.training, config.model, vocab, device)

        # Start training
        console.print(Panel.fit("[bold green]Starting Training[/bold green]"))

        results = trainer.train(
            train_loader, val_loader, resume_from=kwargs.get("resume")
        )

        # Display results
        results_table = Table(title="Training Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="magenta")

        for key, value in results.items():
            if isinstance(value, float):
                value = f"{value:.6f}"
            results_table.add_row(key.replace("_", " ").title(), str(value))

        console.print(results_table)
        console.print("[bold green]Training completed successfully![/bold green]")

    except SLMException as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise click.ClickException(str(e))
    except Exception as e:
        logger.exception("Unexpected error during training")
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Model checkpoint",
)
@click.option("--prompt", "-p", default="", help="Starting prompt")
@click.option("--length", "-l", type=int, default=200, help="Generation length")
@click.option(
    "--temperature", "-t", type=float, default=1.0, help="Sampling temperature"
)
@click.option("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
@click.option("--top-p", type=float, default=0.9, help="Top-p sampling threshold")
@click.option(
    "--sampling",
    type=click.Choice(["greedy", "temperature", "top_k", "top_p"]),
    default="temperature",
    help="Sampling method",
)
@click.option("--seed", type=int, help="Random seed")
@click.option("--device", help="Device to use")
@click.option("--interactive", "-i", is_flag=True, help="Interactive generation mode")
@click.option("--output", "-o", type=click.Path(), help="Save output to file")
@click.pass_context
def generate(ctx, **kwargs):
    """Generate text using a trained model."""
    try:
        # Create generation config
        gen_config = GenerationConfig(
            length=kwargs["length"],
            temperature=kwargs["temperature"],
            top_k=kwargs["top_k"],
            top_p=kwargs["top_p"],
            sampling_method=SamplingMethod(kwargs["sampling"]),
            seed=kwargs.get("seed"),
        )

        # Load generator
        with console.status("[bold green]Loading model..."):
            device = get_device(kwargs.get("device"))
            generator = load_generator_from_checkpoint(
                kwargs["checkpoint"], gen_config, device
            )

        console.print("[green]Model loaded successfully![/green]")

        if kwargs["interactive"]:
            # Interactive mode
            console.print(
                Panel.fit("[bold cyan]Interactive Generation Mode[/bold cyan]")
            )
            console.print(
                "Commands: 'continue' to generate, 'restart' for new prompt, 'quit' to exit"
            )

            result_text = generator.interactive_generate(
                kwargs["prompt"], kwargs["length"]
            )

        else:
            # Single generation
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating text...", total=None)

                def progress_callback(step, total):
                    progress.update(task, completed=step, total=total)

                result = generator.generate(
                    kwargs["prompt"],
                    kwargs["length"],
                    progress_callback=progress_callback,
                )

            # Display results
            console.print(Panel.fit("[bold cyan]Generation Results[/bold cyan]"))

            if kwargs["prompt"]:
                console.print(f"[bold]Prompt:[/bold] {kwargs['prompt']}")
                console.print(f"[bold]Generated Text:[/bold]\n{result['new_text']}")
            else:
                console.print(
                    f"[bold]Generated Text:[/bold]\n{result['generated_text']}"
                )

            # Show stats
            stats_table = Table(title="Generation Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="magenta")

            stats = {
                "Total Tokens": result["total_tokens"],
                "New Tokens": result["new_tokens"],
                "Generation Time": f"{result['generation_time']:.2f}s",
                "Tokens/Second": f"{result['tokens_per_second']:.1f}",
                "Stopped Early": result["stopped_early"],
            }

            for key, value in stats.items():
                stats_table.add_row(key, str(value))

            console.print(stats_table)
            result_text = result["generated_text"]

        # Save to file if requested
        if kwargs.get("output"):
            with open(kwargs["output"], "w", encoding="utf-8") as f:
                f.write(result_text)
            console.print(f"[green]Output saved to: {kwargs['output']}[/green]")

    except SLMException as e:
        console.print(f"[red]Generation failed: {e}[/red]")
        raise click.ClickException(str(e))
    except Exception as e:
        logger.exception("Unexpected error during generation")
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Model checkpoint",
)
@click.pass_context
def info(ctx, checkpoint):
    """Display information about a model checkpoint."""
    try:
        import torch

        with console.status("[bold green]Loading checkpoint..."):
            checkpoint_data = torch.load(checkpoint, map_location="cpu")

        # Model info table
        model_table = Table(title="Model Information")
        model_table.add_column("Property", style="cyan")
        model_table.add_column("Value", style="magenta")

        # Basic info
        basic_info = {
            "Epoch": checkpoint_data.get("epoch", "Unknown"),
            "Loss": f"{checkpoint_data.get('loss', 'Unknown'):.6f}"
            if isinstance(checkpoint_data.get("loss"), float)
            else "Unknown",
            "Best Loss": f"{checkpoint_data.get('best_loss', 'Unknown'):.6f}"
            if isinstance(checkpoint_data.get("best_loss"), float)
            else "Unknown",
            "Is Best": checkpoint_data.get("is_best", "Unknown"),
            "Timestamp": checkpoint_data.get("timestamp", "Unknown"),
        }

        for key, value in basic_info.items():
            model_table.add_row(key, str(value))

        console.print(model_table)

        # Model architecture info
        if "model_info" in checkpoint_data:
            arch_table = Table(title="Model Architecture")
            arch_table.add_column("Property", style="cyan")
            arch_table.add_column("Value", style="magenta")

            model_info = checkpoint_data["model_info"]
            for key, value in model_info.items():
                if key != "config":
                    arch_table.add_row(key.replace("_", " ").title(), str(value))

            console.print(arch_table)

        # Vocabulary info
        if "vocab" in checkpoint_data:
            vocab = checkpoint_data["vocab"]
            vocab_table = Table(title="Vocabulary Information")
            vocab_table.add_column("Property", style="cyan")
            vocab_table.add_column("Value", style="magenta")

            vocab_table.add_row("Size", str(len(vocab)))
            vocab_table.add_row("First 10 chars", str(vocab[:10]))
            vocab_table.add_row("Last 10 chars", str(vocab[-10:]))

            console.print(vocab_table)

    except Exception as e:
        console.print(f"[red]Failed to load checkpoint info: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option("--output", "-o", default="config.yaml", help="Output configuration file")
@click.option("--model-type", type=click.Choice(["rnn", "transformer"]), default="rnn")
@click.pass_context
def init_config(ctx, output, model_type):
    """Create a default configuration file."""
    try:
        config = create_default_config()
        config.model.model_type = ModelType(model_type)

        save_config(config, output)
        console.print(f"[green]Default configuration created: {output}[/green]")
        console.print("Edit this file to customize your training setup.")

    except Exception as e:
        console.print(f"[red]Failed to create config: {e}[/red]")
        raise click.ClickException(str(e))


def train_cli():
    """Entry point for slm-train command."""
    cli(["train"] + sys.argv[1:])


def generate_cli():
    """Entry point for slm-generate command."""
    cli(["generate"] + sys.argv[1:])


if __name__ == "__main__":
    import sys

    cli()
