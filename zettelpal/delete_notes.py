# delete_notes.py - Delete Obsidian Notes by Recording ID

import argparse
import os
import re
import sys

from zettelpal import config


def find_notes_by_recording(vault_path: str, recording_id: str) -> list[str]:
    """
    Finds markdown files with a matching recording ID in frontmatter.
    """
    files_to_delete = []
    pattern = re.compile(rf"^\s*{config.SOURCE_METADATA_KEY}:\s*(.*)", re.MULTILINE)

    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read(2048)
                        match = pattern.search(content)
                        if match:
                            value = match.group(1).strip().strip('"').strip("'")
                            if recording_id in value:
                                files_to_delete.append(filepath)
                except OSError as e:
                    print(f"Warning: Could not read {filepath}: {e}")

    return files_to_delete


def delete_files(file_list: list[str]):
    """Deletes all files in the list."""
    for filepath in file_list:
        try:
            os.remove(filepath)
            print(f"Deleted: {os.path.basename(filepath)}")
        except OSError as e:
            print(f"Error deleting {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Delete Obsidian notes by recording ID."
    )
    parser.add_argument(
        "recording_id",
        help="Recording ID to match (e.g., 11222501)"
    )
    parser.add_argument(
        "--vault",
        default=config.OBSIDIAN_VAULT_ROOT,
        help=f"Vault path (default: {config.OBSIDIAN_VAULT_ROOT})"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    vault_path = os.path.abspath(os.path.expanduser(args.vault))

    if not os.path.exists(vault_path):
        print(f"Error: Vault not found: {vault_path}")
        sys.exit(1)

    print(f"Searching for notes with recording ID '{args.recording_id}'...")
    matching = find_notes_by_recording(vault_path, args.recording_id)

    if not matching:
        print("No matching notes found.")
        sys.exit(0)

    print(f"\nFound {len(matching)} notes to delete:")
    for filepath in matching:
        print(f"  - {os.path.basename(filepath)}")

    if not args.force:
        confirm = input("\nDelete these files? (yes/no): ").lower().strip()
        if confirm != 'yes':
            print("Cancelled.")
            sys.exit(0)

    delete_files(matching)
    print(f"\nDeleted {len(matching)} notes.")


if __name__ == "__main__":
    main()
