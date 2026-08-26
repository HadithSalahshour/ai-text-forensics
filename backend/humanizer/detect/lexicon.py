"""Lexical fingerprints of LLM prose.

Weights are 0-1 and encode *discriminative power*, not frequency. "delve"
carries a near-1.0 weight because its rate in post-2023 machine text is many
times its historical human rate; "important" sits near 0.2 because humans use
it constantly and flagging it would generate false positives.

Everything here is a prior, refit against real data in Phase 3. Treat these
numbers as a starting point, not gospel.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Single-word markers
# --------------------------------------------------------------------------

GPTISM_WEIGHTS: dict[str, float] = {
    # Very high — strongly over-represented in machine text
    "delve": 1.00,
    "delves": 1.00,
    "delving": 1.00,
    "tapestry": 0.95,
    "multifaceted": 0.92,
    "underscore": 0.88,
    "underscores": 0.88,
    "underscoring": 0.88,
    "intricate": 0.85,
    "intricacies": 0.85,
    "pivotal": 0.82,
    "myriad": 0.82,
    "plethora": 0.80,
    "paramount": 0.80,
    "meticulous": 0.80,
    "meticulously": 0.80,
    "nuanced": 0.78,
    "seamless": 0.78,
    "seamlessly": 0.78,
    "realm": 0.75,
    "realms": 0.75,
    "showcase": 0.72,
    "showcases": 0.72,
    "showcasing": 0.72,
    "testament": 0.72,
    "harness": 0.68,
    "harnessing": 0.68,
    "foster": 0.65,
    "fostering": 0.65,
    "bolster": 0.65,
    "bolstering": 0.65,
    "cultivate": 0.62,
    "cultivating": 0.62,
    # Medium — common in human prose but elevated in machine text
    "leverage": 0.58,
    "leveraging": 0.58,
    "landscape": 0.55,
    "navigate": 0.55,
    "navigating": 0.55,
    "robust": 0.52,
    "comprehensive": 0.50,
    "profound": 0.50,
    "profoundly": 0.50,
    "crucial": 0.48,
    "crucially": 0.48,
    "vital": 0.45,
    "invaluable": 0.45,
    "compelling": 0.45,
    "notably": 0.45,
    "arguably": 0.42,
    "essentially": 0.40,
    "fundamentally": 0.40,
    "ultimately": 0.40,
    "holistic": 0.55,
    "synergy": 0.55,
    "transformative": 0.58,
    "unprecedented": 0.50,
    "innovative": 0.42,
    "dynamic": 0.35,
    "significant": 0.30,
    "essential": 0.30,
    "effective": 0.25,
    "numerous": 0.25,
    "various": 0.22,
    "key": 0.18,
    "important": 0.18,
}

# --------------------------------------------------------------------------
# Formulaic multi-word phrases
# --------------------------------------------------------------------------

PHRASE_WEIGHTS: dict[str, float] = {
    r"in today'?s (?:fast[- ]paced|ever[- ]changing|digital|modern) world": 1.00,
    r"the ever[- ]evolving (?:landscape|world|field|nature)": 1.00,
    r"in the (?:realm|world|sphere) of": 0.90,
    r"serves? as a testament to": 0.95,
    r"stands? as a testament to": 0.95,
    r"plays? a (?:crucial|vital|pivotal|significant|key) role": 0.88,
    r"it is important to note that": 0.85,
    r"it'?s worth noting that": 0.80,
    r"a double[- ]edged sword": 0.80,
    r"navigate the complexit(?:y|ies)": 0.85,
    r"delve deeper into": 0.95,
    r"shed(?:s|ding)? light on": 0.62,
    r"pave(?:s|d)? the way for": 0.68,
    r"at the forefront of": 0.62,
    r"a wide (?:range|array) of": 0.50,
    r"when it comes to": 0.48,
    r"in conclusion": 0.62,
    r"overall,? it is clear that": 0.80,
    r"in summary": 0.50,
    r"the importance of .{0,30} cannot be overstated": 0.90,
    r"more than just": 0.55,
    r"unlock(?:ing)? the (?:potential|power|secrets)": 0.85,
    r"embark(?:ing)? on a journey": 0.90,
    r"gain(?:ing)? valuable insights?": 0.75,
    r"tailored to (?:your|their|the) (?:specific )?needs": 0.70,
    r"by understanding .{0,40},? (?:you|we|one) can": 0.70,
    r"whether you'?re a .{0,30} or a": 0.80,
}

_PHRASE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(p, re.IGNORECASE), w) for p, w in PHRASE_WEIGHTS.items()
]

# --------------------------------------------------------------------------
# Sentence-initial discourse connectives
#
# Not wrong — just used at 3-5x the human academic rate, and mechanically
# (one per paragraph, in a predictable rotation).
# --------------------------------------------------------------------------

OPENING_CONNECTIVES: dict[str, float] = {
    "moreover": 0.85,
    "furthermore": 0.85,
    "additionally": 0.80,
    "consequently": 0.70,
    "nevertheless": 0.60,
    "nonetheless": 0.60,
    "in addition": 0.65,
    "however": 0.40,
    "therefore": 0.45,
    "thus": 0.40,
    "indeed": 0.40,
    "notably": 0.55,
    "importantly": 0.62,
    "ultimately": 0.55,
    "overall": 0.50,
    "in essence": 0.75,
    "that said": 0.35,
    "on the other hand": 0.45,
}

# --------------------------------------------------------------------------
# Hedges. The signal is *uniformity* — a flat mid-level hedge on every claim
# with no strong commitments anywhere — not the presence of any one word.
# --------------------------------------------------------------------------

HEDGES: frozenset[str] = frozenset(
    {
        "can", "could", "may", "might", "often", "generally", "typically",
        "usually", "sometimes", "frequently", "largely", "somewhat",
        "relatively", "arguably", "potentially", "possibly", "likely",
        "tends", "tend", "appears", "seems", "suggests", "indicates",
    }
)

BOOSTERS: frozenset[str] = frozenset(
    {
        "must", "always", "never", "certainly", "definitely", "clearly",
        "undoubtedly", "obviously", "proves", "proved", "demonstrates",
        "will", "cannot", "impossible", "guarantee", "guaranteed",
    }
)

# --------------------------------------------------------------------------
# Structural tics
# --------------------------------------------------------------------------

# "It's not just X, it's Y" / "This isn't about X — it's about Y"
NEGATION_PARALLELISM = re.compile(
    r"\b(?:is|are|was|were|it'?s|that'?s|this)\s+n[o']?t\s+(?:just|only|merely|simply|about)\b"
    r"[^.!?]{0,80}?[,;—–-]\s*(?:but|it'?s|they'?re|rather)\b",
    re.IGNORECASE,
)

# Tricolon: "X, Y, and Z" — the rule of three, heavily over-used.
TRICOLON = re.compile(
    r"\b[\w'-]+(?:\s+[\w'-]+){0,2},\s+[\w'-]+(?:\s+[\w'-]+){0,2},\s+and\s+[\w'-]+",
    re.IGNORECASE,
)

# "From X to Y" sweeping-range construction.
FROM_TO_SWEEP = re.compile(r"\bfrom\s+[\w\s'-]{3,30}\s+to\s+[\w\s'-]{3,30}", re.IGNORECASE)

CITATION = re.compile(
    r"\([A-Z][A-Za-z'-]+(?:\s+(?:et al\.?|&|and)\s+[A-Z][A-Za-z'-]+)*,?\s*\d{4}[a-z]?\)"
    r"|\[\d+\]"
)


# --------------------------------------------------------------------------
# Substitutions used by the humanizer (Phase 4)
#
# Register-aware. A naive thesaurus swap produces text that reads wrong and
# that detectors still catch — the replacement has to sit in a believable
# frequency band for the register.
# --------------------------------------------------------------------------

SUBSTITUTIONS: dict[str, dict[str, list[str]]] = {
    "delve into": {
        "academic": ["examine", "investigate", "look closely at"],
        "professional": ["dig into", "go through", "look at"],
        "casual": ["dig into", "get into", "look at"],
    },
    "delve": {
        "academic": ["examine", "investigate"],
        "professional": ["dig", "look"],
        "casual": ["dig", "get"],
    },
    "intricate": {
        "academic": ["complex", "detailed", "elaborate"],
        "professional": ["complicated", "detailed"],
        "casual": ["complicated", "fiddly"],
    },
    "multifaceted": {
        "academic": ["complex", "many-sided"],
        "professional": ["complex", "wide-ranging"],
        "casual": ["complicated"],
    },
    "underscores": {
        "academic": ["highlights", "shows", "points to"],
        "professional": ["shows", "makes clear"],
        "casual": ["shows", "drives home"],
    },
    "underscore": {
        "academic": ["highlight", "show"],
        "professional": ["show", "make clear"],
        "casual": ["show"],
    },
    "pivotal": {
        "academic": ["central", "decisive"],
        "professional": ["central", "make-or-break"],
        "casual": ["big", "make-or-break"],
    },
    "myriad": {
        "academic": ["many", "numerous"],
        "professional": ["many", "a lot of"],
        "casual": ["a lot of", "tons of"],
    },
    "plethora": {
        "academic": ["abundance", "large number"],
        "professional": ["lot", "surplus"],
        "casual": ["ton", "pile"],
    },
    "paramount": {
        "academic": ["critical", "overriding"],
        "professional": ["top priority", "critical"],
        "casual": ["the main thing"],
    },
    "meticulous": {
        "academic": ["careful", "systematic"],
        "professional": ["careful", "thorough"],
        "casual": ["careful", "picky"],
    },
    "nuanced": {
        "academic": ["subtle", "qualified"],
        "professional": ["subtle", "not clear-cut"],
        "casual": ["complicated", "not black and white"],
    },
    "seamless": {
        "academic": ["continuous", "uninterrupted"],
        "professional": ["smooth", "clean"],
        "casual": ["smooth"],
    },
    "realm": {
        "academic": ["field", "domain", "area"],
        "professional": ["area", "space"],
        "casual": ["world", "area"],
    },
    "showcase": {
        "academic": ["demonstrate", "present"],
        "professional": ["show", "present"],
        "casual": ["show off", "show"],
    },
    "harness": {
        "academic": ["use", "apply", "exploit"],
        "professional": ["use", "put to work"],
        "casual": ["use", "tap"],
    },
    "foster": {
        "academic": ["encourage", "promote", "support"],
        "professional": ["encourage", "build"],
        "casual": ["build", "help along"],
    },
    "leverage": {
        "academic": ["use", "draw on", "exploit"],
        "professional": ["use", "draw on"],
        "casual": ["use"],
    },
    "robust": {
        "academic": ["reliable", "well-supported", "sturdy"],
        "professional": ["solid", "reliable"],
        "casual": ["solid", "tough"],
    },
    "comprehensive": {
        "academic": ["thorough", "full", "exhaustive"],
        "professional": ["complete", "full"],
        "casual": ["complete", "full"],
    },
    "crucial": {
        "academic": ["essential", "central"],
        "professional": ["critical", "key"],
        "casual": ["really important"],
    },
    "utilize": {
        "academic": ["use", "employ"],
        "professional": ["use"],
        "casual": ["use"],
    },
    "navigate": {
        "academic": ["work through", "address"],
        "professional": ["work through", "handle"],
        "casual": ["deal with", "get through"],
    },
    "in conclusion": {
        "academic": ["To close", "Taken together"],
        "professional": ["So", "Bottom line"],
        "casual": ["So", "Anyway"],
    },
    "it is important to note that": {
        "academic": ["Note that", "Importantly,"],
        "professional": ["Note that", "Worth flagging:"],
        "casual": ["Worth saying:", "One thing:"],
    },
    "testament": {
        "academic": ["indication", "sign"],
        "professional": ["sign", "proof"],
        "casual": ["sign"],
    },
    "fostering": {
        "academic": ["encouraging", "promoting"],
        "professional": ["building", "encouraging"],
        "casual": ["building"],
    },
    "cultivating": {
        "academic": ["developing", "building"],
        "professional": ["building", "growing"],
        "casual": ["building"],
    },
    "transformative": {
        "academic": ["far-reaching", "substantial"],
        "professional": ["major", "game-changing"],
        "casual": ["huge"],
    },
    "landscape": {
        "academic": ["field", "environment"],
        "professional": ["market", "environment"],
        "casual": ["scene", "world"],
    },
    "profound": {
        "academic": ["substantial", "deep"],
        "professional": ["major", "deep"],
        "casual": ["big", "real"],
    },
    "ultimately": {
        "academic": ["In the end", "Finally"],
        "professional": ["In the end", "Eventually"],
        "casual": ["In the end"],
    },
    "fundamentally": {
        "academic": ["basically", "at root"],
        "professional": ["basically", "at bottom"],
        "casual": ["basically"],
    },
    "significant": {
        "academic": ["notable", "substantial"],
        "professional": ["real", "sizable"],
        "casual": ["big", "real"],
    },
    "bolster": {
        "academic": ["strengthen", "support"],
        "professional": ["strengthen", "shore up"],
        "casual": ["boost"],
    },
    "holistic": {
        "academic": ["overall", "integrated"],
        "professional": ["all-round", "joined-up"],
        "casual": ["all-round"],
    },
    "invaluable": {
        "academic": ["valuable", "essential"],
        "professional": ["very useful", "essential"],
        "casual": ["really useful"],
    },
    "compelling": {
        "academic": ["persuasive", "strong"],
        "professional": ["strong", "convincing"],
        "casual": ["convincing"],
    },
    "unprecedented": {
        "academic": ["novel", "without precedent"],
        "professional": ["new", "first-of-its-kind"],
        "casual": ["brand new"],
    },
    # NOTE: "paramount" is defined earlier in this dict. Do not re-add it here —
    # a duplicate key silently overrides the first definition.
}

# Multi-word formulas whose meaning survives a plain-English replacement.
# Kept apart from SUBSTITUTIONS because these are matched as regexes and
# several of them carry a captured group.
PHRASE_SUBSTITUTIONS: dict[str, dict[str, list[str]]] = {
    r"serves? as a testament to": {
        "academic": ["demonstrates", "shows"],
        "professional": ["shows", "proves"],
        "casual": ["shows"],
    },
    r"stands? as a testament to": {
        "academic": ["demonstrates", "shows"],
        "professional": ["shows"],
        "casual": ["shows"],
    },
    r"the ever[- ]evolving (landscape|world|field|nature) of": {
        "academic": ["the changing {1} of"],
        "professional": ["the shifting {1} of"],
        "casual": ["the changing {1} of"],
    },
    r"the ever[- ]evolving (landscape|world|field|nature)": {
        "academic": ["the changing {1}"],
        "professional": ["the shifting {1}"],
        "casual": ["the changing {1}"],
    },
    r"plays? a (?:crucial|vital|pivotal|significant|key) role in": {
        "academic": ["matters for", "shapes"],
        "professional": ["drives", "shapes"],
        "casual": ["matters for"],
    },
    r"in the (?:realm|world|sphere) of": {
        "academic": ["in", "within"],
        "professional": ["in"],
        "casual": ["in"],
    },
    r"navigate the complexit(?:y|ies) of": {
        "academic": ["work through", "address"],
        "professional": ["work through", "handle"],
        "casual": ["deal with"],
    },
    r"a wide (?:range|array) of": {
        "academic": ["many", "varied"],
        "professional": ["lots of", "many"],
        "casual": ["all sorts of"],
    },
    r"gain(?:ing)? valuable insights?": {
        "academic": ["learning", "finding out"],
        "professional": ["learning", "picking up"],
        "casual": ["learning"],
    },
    r"unlock(?:ing)? the (?:potential|power) of": {
        "academic": ["realizing the potential of"],
        "professional": ["getting value from"],
        "casual": ["getting the most out of"],
    },
    r"pave(?:s|d)? the way for": {
        "academic": ["enables", "makes possible"],
        "professional": ["opens the door to"],
        "casual": ["sets up"],
    },
    r"at the forefront of": {
        "academic": ["leading", "among the leaders in"],
        "professional": ["leading"],
        "casual": ["out front in"],
    },
    r"double[- ]edged sword": {
        "academic": ["mixed blessing", "tradeoff"],
        "professional": ["tradeoff"],
        "casual": ["mixed bag"],
    },
    r"when it comes to": {
        "academic": ["regarding", "for"],
        "professional": ["for", "with"],
        "casual": ["with"],
    },
}


def phrase_hits(text: str) -> list[tuple[str, float, int, int]]:
    """Return (matched_text, weight, start, end) for every formulaic phrase."""
    hits: list[tuple[str, float, int, int]] = []
    for pattern, weight in _PHRASE_PATTERNS:
        for match in pattern.finditer(text):
            hits.append((match.group(0), weight, match.start(), match.end()))
    return hits
