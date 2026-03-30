#!/usr/bin/env python3
"""
separate.py — MetaCog Engine CLI

使い方:
    uv run separate.py input.mp3
    uv run separate.py input.wav --output-dir out/ --max-iter 3 --model htdemucs_ft
    uv run separate.py input.wav --strategy overlap --verbose
"""
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(help="Demucs + Temporal Metacognition Engine")
console = Console()


@app.command()
def separate(
    input_file: Path = typer.Argument(..., help="入力音声ファイル"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", "-o", help="出力ディレクトリ"),
    model: str = typer.Option("htdemucs", "--model", "-m", help="Demucsモデル名"),
    max_iter: int = typer.Option(3, "--max-iter", help="最大イテレーション数"),
    strategy: str = typer.Option("shifts", "--strategy", help="再分離戦略: shifts | overlap"),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
    save_all: bool = typer.Option(False, "--save-all", help="全イテレーションの結果を保存"),
    device: str = typer.Option("auto", "--device", help="cpu | cuda | mps | auto"),
):
    """音声ファイルをステム分離する（メタ認知ループ付き）。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from demucs_metacog.engine import EngineConfig, MetaCogEngine
    from demucs_metacog.io import load_audio, save_stems

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    _device = None if device == "auto" else device

    console.print(Panel(
        f"[bold cyan]MetaCog Stem Separator[/bold cyan]\n"
        f"Input   : {input_file}\n"
        f"Model   : {model}\n"
        f"MaxIter : {max_iter}\n"
        f"Strategy: {strategy}\n"
        f"Output  : {output_dir}",
        title="⚡ Config"
    ))

    # ロード
    with Progress(SpinnerColumn(), TextColumn("[bold green]{task.description}"), transient=True) as prog:
        prog.add_task("Loading audio...", total=None)
        waveform, sr = load_audio(input_file)
    console.print(f"[green]✓[/green] Loaded: {waveform.shape} @ {sr}Hz "
                  f"({waveform.shape[1]/sr:.1f}s)")

    # エンジン初期化
    config = EngineConfig(
        model_name=model,
        device=_device,
        max_iterations=max_iter,
        retry_strategy=strategy,
        verbose=verbose,
    )
    engine = MetaCogEngine(config)

    # 実行
    console.print("\n[bold yellow]Running metacognition loop...[/bold yellow]")
    result = engine.run(waveform, sr)

    # 結果表示
    console.print("\n" + result.summary())

    # 保存
    stem_dir = output_dir / input_file.stem
    saved = save_stems(result.final_stems, stem_dir, result.sample_rate)

    if save_all and result.n_iterations > 1:
        for iter_r in result.iterations[:-1]:
            iter_dir = output_dir / f"{input_file.stem}_iter{iter_r.iteration}"
            save_stems(iter_r.stems, iter_dir, result.sample_rate)
            console.print(f"  Saved iter {iter_r.iteration} → {iter_dir}/")

    console.print(f"\n[bold green]✅ Done![/bold green] Stems saved to: {stem_dir}/")
    for name, path in saved.items():
        console.print(f"  • {name:8s} → {path}")


if __name__ == "__main__":
    app()
