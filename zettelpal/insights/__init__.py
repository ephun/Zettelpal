# insights - Reflection over the vault: theme clustering, resurfacing,
# weekly digests, and ask-yourself retrieval. Everything runs on the local
# backend (embeddings + local LLM); nothing leaves your machines.
#
# Outputs are written as Markdown into the vault's insights subdirectory
# (config.settings.insights_dir), which is excluded from linking and scanning
# so generated notes never mix with your own entries.

from zettelpal.insights.digest import generate_digest
from zettelpal.insights.rag import ask
from zettelpal.insights.resurfacing import resurface
from zettelpal.insights.themes import generate_themes

__all__ = ["generate_themes", "ask", "generate_digest", "resurface"]
