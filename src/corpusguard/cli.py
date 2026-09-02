from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .audit import audit_dataset
from .report import write_html, write_json

app = typer.Typer(
    help="Pre-flight quality and leakage auditing for LLM fine-tuning datasets.",
    no_args_is_help=True,
)

console = Console()


def render_console_report(report) -> None:
    """Render a concise audit report in the terminal."""
    console.print()
    console.print(f"[bold]CorpusGuard[/bold] v{__version__}")
    console.print(f"Dataset: {report.dataset}")
    console.print(
        f"Quality score: [bold]{report.score}/100 ({report.grade})[/bold]"
    )
    console.print()

    metrics = report.metrics.to_dict()

    table = Table(show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    preferred_metrics = [
        ("rows", "rows"),
        ("valid_rows", "valid rows"),
        ("invalid_rows", "invalid rows"),
        ("exact_duplicate_rows", "exact duplicate rows"),
        ("near_duplicate_pairs", "near duplicate pairs"),
        ("pii_hits", "pii hits"),
        ("high_repetition_rows", "high repetition rows"),
        ("leakage_matches", "leakage matches"),
        ("avg_output_chars", "avg output chars"),
        ("p95_output_chars", "p95 output chars"),
    ]

    rendered_keys: set[str] = set()

    for key, label in preferred_metrics:
        if key in metrics:
            table.add_row(label, str(metrics[key]))
            rendered_keys.add(key)

    for key, value in metrics.items():
        if key not in rendered_keys:
            table.add_row(key.replace("_", " "), str(value))

    console.print(table)
    console.print()

    console.print(f"[bold]Findings: {len(report.findings)}[/bold]")

    if not report.findings:
        console.print("No findings.")
    else:
        for finding in report.findings:
            row_text = f" row {finding.row}" if finding.row is not None else ""
            console.print(
                f" • {finding.code}{row_text}: {finding.message}"
            )


@app.command()
def version() -> None:
    """Show CorpusGuard version."""
    typer.echo(__version__)


@app.command()
def scan(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSONL fine-tuning dataset to audit.",
        ),
    ],
    compare: Annotated[
        Path | None,
        typer.Option(
            "--compare",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional second JSONL split to check exact train/eval leakage.",
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json-out",
            help="Write machine-readable report.",
        ),
    ] = None,
    html_out: Annotated[
        Path | None,
        typer.Option(
            "--html-out",
            help="Write standalone HTML report.",
        ),
    ] = None,
    fail_below: Annotated[
        int,
        typer.Option(
            min=0,
            max=100,
            help="Exit 2 if score is below this threshold.",
        ),
    ] = 0,
) -> None:
    """Audit a JSONL fine-tuning dataset."""
    report = audit_dataset(dataset, compare_path=compare)

    render_console_report(report)

    if json_out is not None:
        write_json(report, json_out)
        console.print(f"\nJSON report written to: {json_out}")

    if html_out is not None:
        write_html(report, html_out)
        console.print(f"HTML report written to: {html_out}")

    if fail_below and report.score < fail_below:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()