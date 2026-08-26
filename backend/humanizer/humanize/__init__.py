"""Rewriting: transform library, guards, and the adversarial search loop.

The local engine (Phase 4) needs no API key and no network. The Claude
candidate generator (Phase 5) plugs into the same loop behind the same guards,
so nothing it produces bypasses the meaning and fact checks.
"""

from .loop import Edit, HumanizeResult, humanize
from .transforms import Candidate

__all__ = ["Candidate", "Edit", "HumanizeResult", "humanize"]
