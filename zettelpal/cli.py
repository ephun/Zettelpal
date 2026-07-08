# cli.py - Command-line entry points.
#
# Running with no arguments launches the GUI (so the taskbar-pinned exe just
# works); subcommands cover headless/server use, e.g. from an n8n trigger:
#   zettelpal process 07082601.m4a
#   zettelpal process-dir //server/nextcloud/recordings
#   zettelpal watch //server/nextcloud/recordings --interval 60
#   zettelpal relink --threshold 0.65
#   zettelpal check --fail-on-warning

import os
import sys

import typer

from zettelpal import config
from zettelpal.log import get_logger, setup_console_logging

log = get_logger(__name__)

app = typer.Typer(
    name="zettelpal",
    help="Zettelpal - Privacy-focused audio to Obsidian notes.",
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
)


def _launch_gui():
    # Imported here so headless use never touches tkinter.
    from zettelpal.gui import ZettelpalGUI

    ZettelpalGUI().mainloop()


def _prepare(verbose: bool = False):
    from zettelpal.pipeline import ensure_directories

    setup_console_logging(verbose=verbose)
    ensure_directories()


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug output."),
):
    ctx.obj = {"verbose": verbose}
    if ctx.invoked_subcommand is None:
        _launch_gui()


@app.command()
def process(
    ctx: typer.Context,
    audio_file: str = typer.Argument(..., help="Audio file to process."),
    tags: str = typer.Option("", help="Comma-separated manual tags to add to notes."),
):
    """Intake and process a single recording (renames and dedupes first)."""
    from zettelpal import intake

    _prepare(ctx.obj["verbose"])
    if not os.path.isfile(audio_file):
        log.error(f"File not found: {audio_file}")
        raise typer.Exit(1)
    raise typer.Exit(0 if intake.process_file(audio_file, tags) else 1)


@app.command("process-dir")
def process_dir(
    ctx: typer.Context,
    folder: str = typer.Argument(..., help="Folder containing recordings."),
    tags: str = typer.Option("", help="Comma-separated manual tags to add to notes."),
):
    """One-shot pass over a folder: intake and process every recording.

    Exits 0 if nothing failed, so schedulers (n8n, cron) can alert on failures.
    """
    from zettelpal import intake

    _prepare(ctx.obj["verbose"])
    if not os.path.isdir(folder):
        log.error(f"Folder not found: {folder}")
        raise typer.Exit(1)
    _, _, failed = intake.process_folder(folder, tags)
    raise typer.Exit(1 if failed else 0)


@app.command()
def watch(
    ctx: typer.Context,
    folder: str = typer.Argument(..., help="Folder to monitor (e.g. a Nextcloud sync dir)."),
    interval: int = typer.Option(30, min=1, help="Seconds between folder scans."),
    tags: str = typer.Option("", help="Comma-separated manual tags to add to notes."),
):
    """Continuously monitor a folder and process new recordings as they land."""
    from zettelpal import intake

    _prepare(ctx.obj["verbose"])
    if not os.path.isdir(folder):
        log.error(f"Folder not found: {folder}")
        raise typer.Exit(1)
    try:
        intake.watch_folder(folder, interval=interval, manual_tags=tags)
    except KeyboardInterrupt:
        log.info("[WATCH] Stopped.")


@app.command()
def relink(
    ctx: typer.Context,
    threshold: float = typer.Option(
        None, min=0.0, max=1.0,
        help="Similarity threshold (default: configured value).",
    ),
):
    """Recalculate chronological and semantic links across the whole vault."""
    from zettelpal.vault import linking

    _prepare(ctx.obj["verbose"])
    value = threshold if threshold is not None else config.settings.similarity_threshold
    raise typer.Exit(0 if linking.run_linking_process(value) else 1)


@app.command()
def check(
    ctx: typer.Context,
    vault: str = typer.Option(None, help="Vault root to scan (default: configured vault)."),
    json_report: str = typer.Option(
        "vault_integrity_report.json", "--json", help="Path for the full JSON report."
    ),
    fail_on_warning: bool = typer.Option(
        False, help="Exit non-zero for warnings as well as errors."
    ),
):
    """Run the read-only vault integrity checker."""
    from zettelpal.vault import checker

    setup_console_logging(verbose=ctx.obj["verbose"])
    vault_root = vault or config.settings.vault_root
    raise typer.Exit(checker.run_check(vault_root, json_report, fail_on_warning))


@app.command()
def gui(ctx: typer.Context):
    """Launch the desktop GUI (same as running with no arguments)."""
    _launch_gui()


def main():
    # Compatibility: `zettelpal recording.mp3` still works by rewriting it
    # to `zettelpal process recording.mp3`.
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        sys.argv.insert(1, "process")
    app()


if __name__ == "__main__":
    main()
