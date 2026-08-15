"""The identity layer: who the owner is and what they actually know.

Nothing here is generated. Everything is either something the owner wrote or
something they stated, with evidence attached, which is what makes it safe to
build first-person claims on top of.
"""

from __future__ import annotations

from contentsys.knowledge.ingest import (
    ImportReport,
    add_sample,
    fingerprint,
    import_samples,
    load_seed,
)

__all__ = [
    "ImportReport",
    "add_sample",
    "fingerprint",
    "import_samples",
    "load_seed",
]
