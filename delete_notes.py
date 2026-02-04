import os
import re
import sys
import argparse

def find_files_to_delete(vault_path, search_string):
    """
    Finds markdown files in the specified vault whose 'id' frontmatter property
    contains the search_string.
    """
    files_to_delete = []
    id_pattern = re.compile(r"^\s*id:\s*(.*)", re.MULTILINE)

    for root, dirs, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read(1024)  # Read a small chunk for efficiency
                        match = id_pattern.search(content)
                        if match:
                            file_id = match.group(1).strip().strip('"').strip("'")
                            if search_string in file_id:
                                files_to_delete.append(filepath)
                except Exception as e:
                    print(f"Warning: Could not read file {filepath}. Skipping. Error: {e}")
                    continue
    return files_to_delete

def delete_files(file_list):
    """
    Deletes all files in the provided list.
    """
    for filepath in file_list:
        try:
            os.remove(filepath)
            print(f"Deleted: {filepath}")
        except OSError as e:
            print(f"Error: Could not delete {filepath}. Reason: {e}")

def main():
    parser = argparse.ArgumentParser(description="Delete Obsidian notes based on a search string in their 'id' frontmatter property.")
    parser.add_argument("--vault", type=str, required=True, help="Path to your Obsidian vault.")
    parser.add_argument("--search_string", type=str, required=True, help="The string to match in the 'id' frontmatter property.")
    
    args = parser.parse_args()

    # Normalize and expand the vault path
    vault_path = os.path.abspath(os.path.expanduser(args.vault))

    if not os.path.exists(vault_path):
        print(f"Error: Vault path does not exist: {vault_path}")
        sys.exit(1)

    print(f"Searching for notes in '{vault_path}' with ID containing '{args.search_string}'...")
    matching_files = find_files_to_delete(vault_path, args.search_string)

    if not matching_files:
        print("No matching notes found.")
        sys.exit(0)

    print(f"\nFound {len(matching_files)} notes to delete:")
    for filepath in matching_files:
        print(f"  - {os.path.basename(filepath)}")

    confirm = input("\nAre you sure you want to delete these files? This action is irreversible. (yes/no): ").lower()
    if confirm == 'yes':
        delete_files(matching_files)
        print("\nDeletion complete.")
    else:
        print("\nDeletion cancelled.")

if __name__ == "__main__":
    main()
