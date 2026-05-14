import json
import os
import time
import traceback

import config
import link_notes


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    print(f"[{stamp()}] {message}", flush=True)


def main():
    vault = os.path.abspath(config.OBSIDIAN_VAULT_ROOT)
    threshold = config.SIMILARITY_THRESHOLD

    log("Starting Zettelpal vault link repair")
    log(f"Vault: {vault}")
    log(f"Cache: {config.EMBEDDINGS_CACHE_FILE}")
    log(f"Threshold: {threshold}")

    try:
        log("Step 1/2: scanning vault and loading/updating embeddings")
        started = time.time()

        all_notes, any_updates = link_notes.update_and_load_vault_embeddings(
            vault,
            config.EMBEDDINGS_CACHE_FILE,
            new_notes_info=None,
        )

        log(f"Embedding scan complete in {time.time() - started:.1f}s")
        log(f"Notes with valid embeddings: {len(all_notes)}")
        log(f"Cache updated: {any_updates}")

        log("Step 2/2: forcing full chronological + semantic link rewrite")
        started = time.time()

        link_notes.update_all_notes_links(
            all_notes,
            threshold,
            last_threshold=None,
            any_note_was_updated_or_created=True,
        )

        log(f"Link rewrite complete in {time.time() - started:.1f}s")

        threshold_file = os.path.join(
            vault, ".obsidian", "zettelpal_last_threshold.json"
        )
        with open(threshold_file, "w", encoding="utf-8") as f:
            json.dump({"threshold": threshold}, f, indent=2)

        log(f"Saved threshold file: {threshold_file}")
        log("DONE: vault links rebuilt")
    except Exception:
        log("FAILED")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
