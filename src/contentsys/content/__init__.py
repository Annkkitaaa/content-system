"""The generated layer.

Interpretation built from the identity layer and the external layer. Nothing
here may assert a personal experience that is not backed by a verified row in
``knowledge``, which is enforced in :mod:`contentsys.content.sanitize` rather
than requested in a prompt.
"""

from __future__ import annotations

from contentsys.content.sanitize import (
    SanitizeResult,
    find_experience_claims,
    sanitize,
    strip_banned_punctuation,
)

__all__ = [
    "SanitizeResult",
    "find_experience_claims",
    "sanitize",
    "strip_banned_punctuation",
]
