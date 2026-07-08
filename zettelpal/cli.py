# cli.py - Command-line entry point (launches the GUI when run with no args).

import argparse
import os
import sys

from zettelpal.log import get_logger, setup_console_logging
from zettelpal.pipeline import ensure_directories, run_pipeline

log = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="zettelpal",
        description="Zettelpal - Privacy-focused audio to Obsidian notes"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Audio file to process (launches GUI if not provided)"
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated manual tags to add to notes"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in CLI mode only (requires audio_file)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show debug output"
    )
    args = parser.parse_args()

    setup_console_logging(verbose=args.verbose)
    ensure_directories()

    if args.audio_file:
        # CLI mode
        if not os.path.exists(args.audio_file):
            log.error(f"File not found: {args.audio_file}")
            sys.exit(1)
        success = run_pipeline(args.audio_file, args.tags)
        sys.exit(0 if success else 1)
    elif args.no_gui:
        log.error("--no-gui requires an audio file argument.")
        sys.exit(1)
    else:
        # GUI mode (import here so headless use never touches tkinter)
        from zettelpal.gui import ZettelpalGUI

        app = ZettelpalGUI()
        app.mainloop()


if __name__ == "__main__":
    main()
