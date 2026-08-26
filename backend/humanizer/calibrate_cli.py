"""`humanizer calibrate ...` — corpus management and model training."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import ingest
from .calibration import corpus, fitted
from .calibration import train as train_module

app = typer.Typer(add_completion=False, help="Calibrate the detector on a local corpus.")
console = Console()


@app.command("add")
def add_samples(
    paths: list[str] = typer.Argument(..., help="Files or directories to import."),
    label: str = typer.Option(..., "--label", "-l", help="human | machine"),
    register: str = typer.Option("unknown", "--register", "-r", help=str(corpus.REGISTERS)),
    generator: str = typer.Option("", "--generator", help="Model id, or source note."),
) -> None:
    """Import writing into the calibration corpus.

    Your own writing is the most valuable human data you can add: it teaches the
    detector not to flag the things you personally do. It stays on this machine.
    """
    if label not in corpus.LABELS:
        console.print(f"[red]--label must be one of {corpus.LABELS}[/red]")
        raise typer.Exit(1)

    candidates: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates.extend(
                p for p in sorted(path.rglob("*")) if p.suffix.lower() in ingest.SUPPORTED_SUFFIXES
            )
        else:
            candidates.append(path)

    added = skipped = 0
    for path in candidates:
        try:
            payload = ingest.from_file(path)
        except (FileNotFoundError, ingest.UnsupportedFormatError) as exc:
            console.print(f"[yellow]skip {path.name}: {exc}[/yellow]")
            skipped += 1
            continue

        sample = corpus.add(
            payload["text"],
            label,
            register=register,
            generator=generator or ("" if label == "machine" else "user"),
            topic=path.stem,
        )
        if sample:
            added += 1
            console.print(f"[green]+[/green] {path.name} ({sample.word_count} words)")
        else:
            skipped += 1
            console.print(f"[dim]· {path.name} — too short or already present[/dim]")

    console.print(f"\n[bold]{added} added, {skipped} skipped.[/bold]")


@app.command("generate")
def generate_samples(
    count: int = typer.Option(40, "--count", "-n", help="How many samples to generate."),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Generate machine-text samples locally with the on-device model."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .calibration import generate

    console.print(f"[dim]Generating {count} samples locally — no network, no cost.[/dim]")
    stored = generate.generate(count=count, seed=seed)
    console.print(f"\n[bold]{stored} samples stored.[/bold]")


@app.command("status")
def status() -> None:
    """Show corpus composition and any problems with it."""
    info = corpus.summary()

    if info["total"] == 0:
        console.print("[yellow]Corpus is empty.[/yellow]")
        console.print(
            "\nAdd machine text:  [dim]humanizer calibrate generate -n 40[/dim]"
            "\nAdd your writing:  [dim]humanizer calibrate add ~/essays -l human[/dim]"
        )
        return

    table = Table(show_edge=False, header_style="dim")
    table.add_column("Label")
    table.add_column("Count", justify="right")
    for label, count in sorted(info["by_label"].items()):
        table.add_row(label, str(count))
    table.add_row("[dim]words[/dim]", f"[dim]{info['total_words']:,}[/dim]")
    console.print(Panel(table, title="Corpus"))

    if info["by_register"]:
        registers = Table(show_edge=False, header_style="dim", title="By register")
        registers.add_column("Label / register")
        registers.add_column("Count", justify="right")
        for key, count in sorted(info["by_register"].items()):
            registers.add_row(key, str(count))
        console.print(registers)

    issues = corpus.audit()
    if issues:
        console.print("\n[bold]Issues[/bold]")
        for issue in issues:
            color = "red" if issue.severity == "blocker" else "yellow"
            console.print(f"  [{color}]{issue.severity}[/{color}]  {issue.message}")
    else:
        console.print("\n[green]No issues found.[/green]")


@app.command("train")
def train_model(
    no_models: bool = typer.Option(False, "--no-models", help="Skip the statistical tier."),
    no_save: bool = typer.Option(False, "--no-save", help="Evaluate without saving."),
) -> None:
    """Fit the detector on the corpus and report cross-validated metrics."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        report = train_module.train(use_models=not no_models, save=not no_save)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    m = report.metrics
    table = Table(show_edge=False, header_style="dim")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Samples", f"{m.n_samples} ({m.n_human} human / {m.n_machine} machine)")
    table.add_row("AUROC", f"{m.auroc_mean:.3f} ± {m.auroc_std:.3f}")
    table.add_row("Accuracy", f"{m.accuracy:.3f}")
    label = f"TPR @ {m.target_fpr:.1%} FPR"
    if not m.fpr_resolvable:
        label += " [dim](coarser than 1% — see below)[/dim]"
    table.add_row(label, f"{m.tpr_at_target_fpr:.3f}")

    verdict = (
        "[green]usable[/green]" if report.trustworthy else "[yellow]not yet trustworthy[/yellow]"
    )
    console.print(Panel(table, title=f"Calibration — {verdict}"))

    if report.feature_importance:
        importance = Table(show_edge=False, header_style="dim", title="Most informative features")
        importance.add_column("Feature")
        importance.add_column("Δ AUROC when shuffled", justify="right")
        for name, value in report.feature_importance[:12]:
            importance.add_row(name, f"{value:+.4f}")
        console.print(importance)

    for warning in report.warnings:
        color = "red" if warning.startswith("[blocker]") else "yellow"
        console.print(f"[{color}]{warning}[/{color}]")

    if not report.trustworthy:
        console.print(
            "\n[dim]The ensemble will keep using its hand-tuned weights until a "
            "training run clears the bar.[/dim]"
        )
    fitted.reset_cache()


@app.command("show")
def show_model() -> None:
    """Show the currently loaded fitted model, if any."""
    model = fitted.load()
    if model is None:
        console.print("[yellow]No fitted model. Run `humanizer calibrate train`.[/yellow]")
        return
    state = "[green]in use[/green]" if model.trustworthy else "[yellow]ignored (below bar)[/yellow]"
    console.print(f"{model.summary}  —  {state}")
