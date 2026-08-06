"""The analyst: turn the brief + evidence pile into a proposal.

This is where the engine "thinks": it orders the evidence following the brief's
argument, writes each exhibit's cover-page text (what it proves), paginates, and
plans in-brief citations. The deterministic pipeline always works with no API
key; when a CrewAI crew is configured (``crew.write_covers``), its CoverWriter
agent produces the human-authored cover text instead.

Everything downstream — covers, index, packet, citations — is derived from the
``ExhibitSet`` this module builds, so any late insertion renumbers every artifact
that depends on position.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .citations import _keywords, plan_citations
from .model import Citation, Exhibit, ExhibitSet
from .order import EvidenceItem, propose_order
from .paging import paginate_set

# A cover writer is (title, content) -> "what it proves" sentence.
CoverWriter = Callable[[str, str], str]


@dataclass
class Proposal:
    """The analyst's output: an ordered, paginated exhibit set and its citations."""

    set: ExhibitSet
    citations: list[Citation]
    order_note: str

    def repropose(self, brief: str) -> "Proposal":
        """After the set changed (e.g. a late insertion renumbered it), recompute
        everything derived from position — page numbers and in-brief citations —
        and update this proposal in place. Returns self for chaining."""
        paginate_set(self.set)
        self.citations = plan_citations(self.set, brief)
        self.order_note = "Re-paginated and re-cited after the set changed."
        return self


def propose(brief: str, items: list[EvidenceItem],
            cover_writer: CoverWriter | None = None) -> Proposal:
    """Build a full proposal from the brief and the evidence pile."""
    ordered = propose_order(brief, items)
    writer = cover_writer or _default_cover_writer

    set_ = ExhibitSet()
    for item in ordered:
        title = _human_title(item.source)
        set_.add(
            source=Path(item.source),
            title=title,
            proves=writer(title, item.text),
            content=item.text,
        )
    paginate_set(set_)
    return Proposal(
        set=set_,
        citations=plan_citations(set_, brief),
        order_note=_order_note(ordered),
    )


def replan(set_: ExhibitSet, brief: str) -> Proposal:
    """Compatibility wrapper around :meth:`Proposal.repropose` for callers that
    want a fresh Proposal object. Prefer ``proposal.repropose(brief)``."""
    return Proposal(set=set_, citations=[], order_note="").repropose(brief)


def _human_title(source: str) -> str:
    base = Path(source).name  # strip any directory prefix
    base = re.sub(r"\.(pdf|docx?|md|txt|html?|rtf)$", "", base, flags=re.I)
    base = re.sub(r"[^A-Za-z0-9 ]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return " ".join(w.capitalize() for w in base.split()[:8]) or "Exhibit"


def _default_cover_writer(title: str, content: str) -> str:
    """Deterministic cover text from the exhibit's own content.

    Names and places (capitalised words in the source) are excluded so the line
    reads like a paralegal's, grounded in the document rather than listing
    proper nouns. A CrewAI CoverWriter agent replaces this when configured.
    """
    from .citations import _STOP

    from collections import Counter as _Counter
    from .citations import _evidence_content_keywords

    freq = _Counter(_evidence_content_keywords(content))
    top = [w for w, _ in freq.most_common(3) if w]
    if not top:
        return f"{title} supports the claims made in this application."
    return f"Evidences {top[0]}, {top[1]}, and {top[2]} relevant to this application."


def _order_note(ordered: list[EvidenceItem]) -> str:
    return (
        f"Ordered {len(ordered)} pieces of evidence to follow the brief's "
        "argument; evidence not touched by the brief is appended last."
    )
