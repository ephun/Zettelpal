# cli.py - Command-line entry point (launches the GUI when run with no args).

import argparse
import os
import sys

from zettelpal.pipeline import ensure_directories, run_pipeline


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
    args = parser.parse_args()

    ensure_directories()

    if args.audio_file:
        # CLI mode
        if not os.path.exists(args.audio_file):
            print(f"ERROR: File not found: {args.audio_file}")
            sys.exit(1)
        success = run_pipeline(args.audio_file, args.tags)
        sys.exit(0 if success else 1)
    elif args.no_gui:
        print("ERROR: --no-gui requires an audio file argument.")
        sys.exit(1)
    else:
        # GUI mode (import here so headless use never touches tkinter)
        from zettelpal.gui import ZettelpalGUI

        app = ZettelpalGUI()
        app.mainloop()


if __name__ == "__main__":
    main()
