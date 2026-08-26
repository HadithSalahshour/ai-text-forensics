"""Command-line interface.

    humanizer score essay.docx
    humanizer score --text "Some pasted prose..."
    humanizer score essay.md --sentences
    humanizer info
"""

from __future__ import annotations

import sys
from pathlib import Path  # noqa: F401  (used by the humanize command)

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import ingest, segment
from .config import settings
from .detect import ensemble, rulescore
from .nlp import has_nlp

app = typer.Typer(add_completion=False, help="Local AI-writing detector and humanizer.")
console = Console()

from .calibrate_cli import app as calibrate_app  # noqa: E402

app.add_typer(calibrate_app, name="calibrate")


def _verdict(score: float) -> tuple[str, str]:
    if score < 20:
        return "Reads as human", "green"
    if score < 40:
        return "Mostly human", "bright_green"
    if score < 60:
        return "Mixed signals", "yellow"
    if score < 80:
        return "Likely AI-generated", "orange3"
    return "Very likely AI-generated", "red"


def _bar(fraction: float, width: int = 24) -> str:
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


@app.command()
def score(
    path: str = typer.Argument(None, help="Path to .txt/.md/.docx/.pdf"),
    text: str = typer.Option(None, "--text", "-t", help="Score literal text instead."),
    sentences: bool = typer.Option(False, "--sentences", "-s", help="Per-sentence heatmap."),
    limit: int = typer.Option(12, "--limit", help="Max sentences to list."),
    fast: bool = typer.Option(
        False, "--fast", help="Skip the local models; style signals only."
    ),
) -> None:
    """Analyze a document and report its AI-writing score."""
    if text:
        payload = ingest.from_text(text)
    elif path:
        try:
            payload = ingest.from_file(path)
        except (FileNotFoundError, ingest.UnsupportedFormatError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
    else:
        if sys.stdin.isatty():
            console.print("[red]Provide a file path, --text, or pipe text on stdin.[/red]")
            raise typer.Exit(1)
        payload = ingest.from_text(sys.stdin.read())

    if not payload["text"].strip():
        console.print("[red]No readable text found.[/red]")
        raise typer.Exit(1)

    doc = segment.build(**payload)

    if not fast:
        with console.status("[dim]Loading local models...[/dim]"):
            result = ensemble.analyze(doc, use_models=True)
    else:
        result = ensemble.analyze(doc, use_models=False)

    label, color = _verdict(result.score)
    header = Text.assemble(
        (f"{result.score:.1f}", f"bold {color}"),
        ("/100  ", "dim"),
        (label, color),
    )
    console.print(
        Panel(
            header,
            title=f"[dim]{doc.source_name}[/dim]",
            subtitle=(
                f"[dim]{doc.word_count} words · {len(doc.sentences)} sentences · "
                f"{len(doc.paragraphs)} paragraphs[/dim]"
            ),
        )
    )

    if result.note:
        console.print(f"[yellow]⚠  {result.note}[/yellow]\n")

    # Per-tier breakdown, so it's obvious whether the verdict rests on
    # sampling signature or on writing style.
    tiers = Table(title="By tier", show_edge=False, header_style="dim")
    tiers.add_column("Tier")
    tiers.add_column("Score", justify="right")
    tiers.add_column("")
    tier_labels = {
        "statistical": "Statistical (model-based)",
        "structure": "Structure",
        "lexicon": "Lexicon",
        "mechanics": "Mechanics",
        "syntax": "Syntax",
    }
    for key, tier_label in tier_labels.items():
        if key not in result.tier_scores:
            continue
        value = result.tier_scores[key]
        shade = "red" if value > 66 else "yellow" if value > 33 else "green"
        tiers.add_row(tier_label, f"{value:.1f}", f"[{shade}]{_bar(value / 100)}[/{shade}]")
    console.print(tiers)

    table = Table(title="Strongest signals", show_edge=False, header_style="dim")
    table.add_column("Signal")
    table.add_column("Tier", style="dim")
    table.add_column("Value", justify="right")
    table.add_column("Contribution", justify="left")

    for signal in result.top_signals(10):
        shade = "red" if signal.score > 0.66 else "yellow" if signal.score > 0.33 else "green"
        table.add_row(
            signal.label,
            signal.tier,
            f"{signal.value:.2f}",
            f"[{shade}]{_bar(signal.score)}[/{shade}] {signal.score * 100:>3.0f}%",
        )
    console.print(table)

    if not has_nlp():
        console.print(
            f"\n[yellow]Syntactic signals are disabled — install the spaCy model:[/yellow]\n"
            f"  [dim]python -m spacy download {settings.spacy_model}[/dim]"
        )
    if not fast and not result.statistical_available and result.statistical_error:
        console.print(f"\n[yellow]{result.statistical_error}[/yellow]")

    if sentences:
        console.print("\n[bold]Sentence heatmap[/bold]  [dim](highest-scoring first)[/dim]")
        scored = rulescore.score_sentences(doc)
        by_sentence = {s.index: s for s in doc.sentences}
        for index, sent_score, reasons in sorted(scored, key=lambda r: -r[1])[:limit]:
            shade = "red" if sent_score > 66 else "yellow" if sent_score > 33 else "green"
            snippet = by_sentence[index].text
            if len(snippet) > 110:
                snippet = snippet[:107] + "..."
            console.print(f"\n  [{shade}]{sent_score:>5.1f}[/{shade}]  {snippet}")
            for reason in reasons:
                console.print(f"         [dim]· {reason}[/dim]")


@app.command()
def humanize(
    path: str = typer.Argument(None, help="Path to .txt/.md/.docx/.pdf"),
    text: str = typer.Option(None, "--text", "-t", help="Rewrite literal text."),
    strength: str = typer.Option("moderate", "--strength", help="light | moderate | aggressive"),
    register: str = typer.Option(None, "--register", help="Force academic/professional/casual."),
    out: str = typer.Option(None, "--out", "-o", help="Write the result to a file."),
    show_diff: bool = typer.Option(False, "--diff", "-d", help="Show every edit."),
    claude: bool = typer.Option(
        False, "--claude", help="Also use Claude. Sends this document to the API."
    ),
    high_quality: bool = typer.Option(
        False, "--high-quality", help="Escalate the Claude model (costs more)."
    ),
    budget: float = typer.Option(
        0.15, "--budget", help="Hard cap on API spend for this document, in USD."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost confirmation."),
) -> None:
    """Rewrite a document so it reads as human, without changing what it says."""
    from .humanize import humanize as run_humanize

    if text:
        payload = ingest.from_text(text)
    elif path:
        try:
            payload = ingest.from_file(path)
        except (FileNotFoundError, ingest.UnsupportedFormatError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
    else:
        if sys.stdin.isatty():
            console.print("[red]Provide a file path, --text, or pipe text on stdin.[/red]")
            raise typer.Exit(1)
        payload = ingest.from_text(sys.stdin.read())

    doc = segment.build(**payload)

    # Confirm before spending money or sending text off the machine. Both are
    # things the user should opt into per document, not discover afterwards.
    if claude:
        from .humanize.llm import ClaudeRewriter

        rewriter = ClaudeRewriter(
            model=settings.quality_model if high_quality else settings.bulk_model
        )
        status = rewriter.status()
        if not status.available:
            console.print(f"[yellow]Claude unavailable: {status.reason}[/yellow]")
            console.print("[dim]Continuing with the local engine only.[/dim]\n")
            claude = False
        elif not yes:
            from .humanize.loop import STRENGTH_PRESETS

            preset = STRENGTH_PRESETS.get(strength, STRENGTH_PRESETS["moderate"])
            max_calls = preset["llm_paragraphs"] * preset["max_iterations"]
            estimate = min(rewriter.estimate_cost(doc.text, calls=max_calls), budget)
            console.print(
                f"[yellow]This will send {doc.word_count} words to the Anthropic API "
                f"using {rewriter.model}.[/yellow]\n"
                f"[dim]Up to {max_calls} calls at '{strength}' strength · "
                f"estimated ~${estimate:.3f} · hard cap ${budget:.2f}.[/dim]"
            )
            if not typer.confirm("Continue?", default=True):
                raise typer.Abort()

    with console.status("[dim]Rewriting...[/dim]"):
        result = run_humanize(
            doc,
            strength=strength,
            register=register,
            use_llm=claude,
            high_quality=high_quality,
            budget_usd=budget if claude else None,
        )

    before_label, before_color = _verdict(result.score_before)
    after_label, after_color = _verdict(result.score_after)
    console.print(
        Panel(
            Text.assemble(
                (f"{result.score_before:.1f}", f"bold {before_color}"),
                (f"  {before_label}", before_color),
                ("   →   ", "dim"),
                (f"{result.score_after:.1f}", f"bold {after_color}"),
                (f"  {after_label}", after_color),
            ),
            title=f"[dim]{doc.source_name}[/dim]",
            subtitle=(
                f"[dim]{len(result.edits)} edits · meaning preserved "
                f"{(result.document_similarity or 1.0):.3f}[/dim]"
            ),
        )
    )

    for note in result.notes:
        style = "yellow" if ("WARNING" in note or "revert" in note) else "dim"
        console.print(f"[{style}]· {note}[/{style}]")

    if result.llm_usage_summary:
        console.print(f"[dim]· API usage: {result.llm_usage_summary}[/dim]")

    if show_diff:
        console.print("\n[bold]Edits[/bold]")
        for edit in result.edits:
            console.print(f"\n  [dim]{edit.transform}[/dim]")
            console.print(f"  [red]- {edit.original}[/red]")
            console.print(f"  [green]+ {edit.rewritten}[/green]")

    if out:
        Path(out).write_text(result.final_text, encoding="utf-8")
        console.print(f"\n[green]Written to {out}[/green]")
    elif not show_diff:
        console.print("\n" + result.final_text)


@app.command()
def info() -> None:
    """Show configuration and component availability."""
    table = Table(show_edge=False, header_style="dim")
    table.add_column("Setting")
    table.add_column("Value")

    network = (
        "[yellow]Claude API enabled[/yellow]"
        if settings.llm_enabled
        else "[green]Local only — no network[/green]"
    )
    table.add_row("Privacy mode", network)
    table.add_row("Server", f"{settings.host}:{settings.port}")
    table.add_row(
        "spaCy model",
        f"[green]{settings.spacy_model} ready[/green]"
        if has_nlp()
        else f"[yellow]{settings.spacy_model} not installed[/yellow]",
    )
    table.add_row("Scorer model", f"{settings.scorer_model} [dim](Phase 2)[/dim]")
    table.add_row("Reference model", f"{settings.reference_model} [dim](Phase 2)[/dim]")
    table.add_row("Rewrite model", f"{settings.bulk_model} [dim](Phase 5)[/dim]")
    table.add_row("Target score", f"< {settings.target_score:.0f}")
    table.add_row("Model cache", str(settings.model_dir))
    table.add_row("Data dir", str(settings.data_dir))

    console.print(Panel(table, title="humanizer"))


@app.command()
def serve() -> None:
    """Start the local web UI server."""
    from .api import serve as run_server

    mode = (
        "[yellow]Claude API enabled[/yellow]"
        if settings.llm_enabled
        else "[green]Local only[/green]"
    )
    console.print(
        Panel(
            f"http://{settings.host}:{settings.port}\n\n{mode}",
            title="humanizer server",
            subtitle="[dim]loopback only — not reachable from your network[/dim]",
        )
    )
    run_server()


@app.command()
def formats() -> None:
    """List supported input formats."""
    console.print(", ".join(sorted(ingest.SUPPORTED_SUFFIXES)))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
