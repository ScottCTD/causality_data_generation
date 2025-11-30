"""
Option semantics and available ``OptionFact.kind`` values.

This module defines a small, closed vocabulary of logical facts about a single
ball's outcome in a shot.  Each fact is represented as an ``OptionFact`` with
two pieces of information:

  * ``kind`` – a short string identifying the predicate.
  * ``args`` – a tuple of positional arguments (if any) parameterizing it.

The renderer in ``OptionRenderer`` turns these facts into human‑readable
strings in a given ``Tense`` (base / future / conditional).  All multiple‑choice
options are ultimately derived from these facts.

Current kinds
=============

Pocket‑related
--------------
* ``"pocketed"``:
    - Args: ``()`` (no arguments).
    - Meaning: The cue ball ends up in *some* pocket.
    - Example renderings:
        - BASE: ``"The ball was pocketed"``
        - FUTURE: ``"The ball will be pocketed"``
        - CONDITIONAL: ``"The ball would be pocketed"``.

* ``"pocketed_in"``:
    - Args: ``(pocket_color: str,)``.
    - ``pocket_color`` is a string such as one of ``POCKET_COLORS``:
      ``"gray"``, ``"purple"``, ``"blue"``, ``"orange"``, ``"green"``, ``"red"``.
    - Meaning: The cue ball ends up specifically in the given pocket.
    - Example renderings:
        - BASE: ``"The ball was pocketed in the gray pocket"``
        - FUTURE: ``"The ball will be pocketed in the gray pocket"``
        - CONDITIONAL: ``"The ball would be pocketed in the gray pocket"``.

* ``"not_pocketed"``:
    - Args: ``()``.
    - Meaning: The cue ball is *not* pocketed by the end of the shot.
    - Example renderings:
        - BASE: ``"The ball was not pocketed"``
        - FUTURE: ``"The ball will not be pocketed"``
        - CONDITIONAL: ``"The ball would not be pocketed"``.

Wall‑count / aggregate bounce facts
-----------------------------------
These describe how many walls were contacted, but not which specific walls.

* ``"hits_0_walls"``:
    - Args: ``()``.
    - Meaning: The cue ball does not hit any walls at all.
    - Example: ``"The ball hits 0 walls"`` (tense‑adapted by ``OptionRenderer``).

* ``"hits_1_wall"``:
    - Args: ``()``.
    - Meaning: The cue ball hits exactly one wall (at least once).
    - Example: ``"The ball hits 1 wall"``.

* ``"hits_n_diff_walls"``:
    - Args: ``(n: int,)`` where ``n >= 2``.
    - Meaning: The cue ball hits ``n`` *distinct* walls in total.
    - Example (``n=3``): ``"The ball hits 3 different walls"``.

* ``"hits_same_wall_n_times"``:
    - Args: ``(n: int,)`` where ``n >= 2``.
    - Meaning: Every recorded wall contact is with the *same* wall, exactly
      ``n`` times in total.
    - Example (``n=3``): ``"The ball hits the same wall 3 times"``.

Wall‑sequence facts
-------------------
These describe the *order* and identity of early wall contacts.
Wall names are strings such as those in ``WALL_NAMES``:
``"green-blue-wall"``, ``"orange-red-wall"``, ``"grey-orange-wall"``,
``"purple-grey-wall"``, ``"blue-purple-wall"``, ``"red-green-wall"``.

* ``"first_wall_hit"``:
    - Args: ``(wall_name: str,)``.
    - Meaning: The first wall that the cue ball hits has the given name.
    - Example: ``"The first wall hit was green-blue-wall"``.

* ``"second_wall_hit"``:
    - Args: ``(wall_name: str,)``.
    - Meaning: The second wall that the cue ball hits has the given name.
    - Example: ``"The second wall hit was purple-grey-wall"``.

* ``"third_wall_hit"``:
    - Args: ``(wall_name: str,)``.
    - Meaning: The third wall that the cue ball hits has the given name.
    - Example: ``"The third wall hit was red-green-wall"``.

Extensibility
=============
To introduce a new kind of option (for example, statements about ball‑ball
collisions or whether the ball ever reverses direction), you should:

1. Add a new ``kind`` name and argument convention to this docstring.
2. Teach ``facts_from_outcome`` how to emit the corresponding ``OptionFact``.
3. Extend ``OptionRenderer.render`` (and, if needed, helper methods) so the
   new fact can be phrased in all supported tenses.
4. Optionally include the new ``OptionFact`` in ``DISTRACTOR_POOL_FACTS`` so
   it can be used as a distractor when not true in a given scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from .tense import Tense
except ImportError:  # script-level import
    from tense import Tense  # type: ignore


# Canonical pocket colors and wall names used in the distractor pool.
POCKET_COLORS = ["gray", "purple", "blue", "orange", "green", "red"]
WALL_NAMES = [
    "green-blue-wall",
    "orange-red-wall",
    "grey-orange-wall",
    "purple-grey-wall",
    "blue-purple-wall",
    "red-green-wall",
]


@dataclass(frozen=True)
class OptionFact:
    """
    Structured representation of a single answer option, independent of tense.

    ``kind`` identifies the logical predicate (e.g., ``\"pocketed\"``), while
    ``args`` carry any parameters such as pocket color, wall name, or counts.
    """

    kind: str
    args: Tuple[object, ...] = ()


class OptionRenderer:
    """
    Responsible for turning ``OptionFact`` instances into human-readable strings
    in a given tense. This centralizes all phrasing decisions so that adding new
    tenses or rewording options does not affect the rest of the pipeline.
    """

    def render(self, fact: OptionFact, tense: Tense) -> str:
        k = fact.kind
        args = fact.args

        if k == "pocketed":
            return self._pocketed(tense)
        if k == "pocketed_in":
            color = args[0]
            return self._pocketed_in(color, tense)
        if k == "not_pocketed":
            return self._not_pocketed(tense)

        if k == "hits_0_walls":
            return self._hits_n_walls(0, tense)
        if k == "hits_1_wall":
            return self._hits_n_walls(1, tense)
        if k == "hits_n_diff_walls":
            n = int(args[0])
            return self._hits_diff_walls(n, tense)
        if k == "hits_same_wall_n_times":
            n = int(args[0])
            return self._hits_same_wall_n_times(n, tense)

        if k == "first_wall_hit":
            name = args[0]
            return self._wall_hit_order(1, name, tense)
        if k == "second_wall_hit":
            name = args[0]
            return self._wall_hit_order(2, name, tense)
        if k == "third_wall_hit":
            name = args[0]
            return self._wall_hit_order(3, name, tense)

        # Fallback to a simple string representation that includes kind/args.
        return f"{k}({', '.join(map(str, args))})"

    # --- Pocket helpers -------------------------------------------------
    @staticmethod
    def _pocketed(tense: Tense) -> str:
        if tense is Tense.FUTURE:
            return "The ball will be pocketed"
        if tense is Tense.CONDITIONAL:
            return "The ball would be pocketed"
        return "The ball was pocketed"

    @staticmethod
    def _pocketed_in(color: str, tense: Tense) -> str:
        if tense is Tense.FUTURE:
            return f"The ball will be pocketed in the {color} pocket"
        if tense is Tense.CONDITIONAL:
            return f"The ball would be pocketed in the {color} pocket"
        return f"The ball was pocketed in the {color} pocket"

    @staticmethod
    def _not_pocketed(tense: Tense) -> str:
        if tense is Tense.FUTURE:
            return "The ball will not be pocketed"
        if tense is Tense.CONDITIONAL:
            return "The ball would not be pocketed"
        return "The ball was not pocketed"

    # --- Wall-hit helpers -----------------------------------------------
    @staticmethod
    def _hits_n_walls(n: int, tense: Tense) -> str:
        # n should be 0 or 1 in current usage, but the format is generic.
        unit = "wall" if n == 1 else "walls"
        if tense is Tense.FUTURE:
            verb = "will hit"
        elif tense is Tense.CONDITIONAL:
            verb = "would hit"
        else:
            verb = "hits"
        return f"The ball {verb} {n} {unit}"

    @staticmethod
    def _hits_diff_walls(n: int, tense: Tense) -> str:
        if tense is Tense.FUTURE:
            verb = "will hit"
        elif tense is Tense.CONDITIONAL:
            verb = "would hit"
        else:
            verb = "hits"
        return f"The ball {verb} {n} different walls"

    @staticmethod
    def _hits_same_wall_n_times(n: int, tense: Tense) -> str:
        if tense is Tense.FUTURE:
            verb = "will hit"
        elif tense is Tense.CONDITIONAL:
            verb = "would hit"
        else:
            verb = "hits"
        return f"The ball {verb} the same wall {n} times"

    @staticmethod
    def _wall_hit_order(order: int, name: str, tense: Tense) -> str:
        if order == 1:
            prefix = "first"
        elif order == 2:
            prefix = "second"
        else:
            prefix = "third"

        if tense is Tense.FUTURE:
            mid = "hit will be"
        elif tense is Tense.CONDITIONAL:
            mid = "hit would be"
        else:
            mid = "hit was"
        return f"The {prefix} wall {mid} {name}"


def facts_from_outcome(outcomes: Dict) -> List[OptionFact]:
    """
    Convert a normalized single-ball outcomes dict into a set of logical facts.

    This is the only place that peeks into the structure of the outcomes dict.
    Everything downstream operates at the ``OptionFact`` level.
    """
    facts: List[OptionFact] = []

    hits = int(outcomes.get("num_wall_hits", 0) or 0)
    wall_hits = list(outcomes.get("wall_hits", []) or [])
    pocketed = bool(outcomes.get("pocketed", False))
    pocket_color = outcomes.get("pocket_color")

    # Pocket-related facts
    if pocketed:
        facts.append(OptionFact("pocketed"))
        if pocket_color:
            facts.append(OptionFact("pocketed_in", (str(pocket_color),)))
    else:
        facts.append(OptionFact("not_pocketed"))

    # Aggregate wall-hit count facts
    if hits == 0:
        facts.append(OptionFact("hits_0_walls"))
    elif hits == 1:
        facts.append(OptionFact("hits_1_wall"))
    elif hits >= 2:
        # If all wall hits are the same label, we can say "same wall N times".
        if wall_hits and len(wall_hits) == hits and len(set(wall_hits)) == 1:
            facts.append(OptionFact("hits_same_wall_n_times", (hits,)))
        else:
            facts.append(OptionFact("hits_n_diff_walls", (hits,)))

    # Sequence-based wall facts (first / second / third hit labels).
    if wall_hits:
        facts.append(OptionFact("first_wall_hit", (wall_hits[0],)))
        if len(wall_hits) > 1:
            facts.append(OptionFact("second_wall_hit", (wall_hits[1],)))
        if len(wall_hits) > 2:
            facts.append(OptionFact("third_wall_hit", (wall_hits[2],)))

    # Deduplicate while preserving order.
    seen = set()
    unique_facts: List[OptionFact] = []
    for fact in facts:
        if fact in seen:
            continue
        seen.add(fact)
        unique_facts.append(fact)
    return unique_facts


def _describes_pocketed(fact: OptionFact) -> bool:
    return fact.kind in {"pocketed", "pocketed_in"}


def _describes_not_pocketed(fact: OptionFact) -> bool:
    return fact.kind == "not_pocketed"


def _describes_zero_wall_hits(fact: OptionFact) -> bool:
    return fact.kind == "hits_0_walls"


def _describes_positive_wall_hits(fact: OptionFact) -> bool:
    return fact.kind in {
        "hits_1_wall",
        "hits_n_diff_walls",
        "hits_same_wall_n_times",
        "first_wall_hit",
        "second_wall_hit",
        "third_wall_hit",
    }


def _is_consistent_distractor(candidate: OptionFact, chosen_true: Sequence[OptionFact]) -> bool:
    """
    Decide whether a candidate distractor fact is logically compatible with
    the already-chosen true facts. This operates on structured facts rather
    than brittle string-based heuristics.
    """
    for fact in chosen_true:
        # Pocketed vs not-pocketed are mutually exclusive.
        if _describes_pocketed(fact) and _describes_not_pocketed(candidate):
            return False
        if _describes_not_pocketed(fact) and _describes_pocketed(candidate):
            return False

        # Zero wall hits vs any wall-hit descriptions are mutually exclusive.
        if _describes_zero_wall_hits(fact) and _describes_positive_wall_hits(candidate):
            return False
        if _describes_zero_wall_hits(candidate) and _describes_positive_wall_hits(fact):
            return False

    return True


# Build a global pool of possible distractor facts. This mirrors the original
# OPTION_POOL strings but expressed in terms of OptionFact semantics.
DISTRACTOR_POOL_FACTS: List[OptionFact] = []

# Pocket-related distractors.
DISTRACTOR_POOL_FACTS.append(OptionFact("pocketed"))
for color in POCKET_COLORS:
    DISTRACTOR_POOL_FACTS.append(OptionFact("pocketed_in", (color,)))
DISTRACTOR_POOL_FACTS.append(OptionFact("not_pocketed"))

# Wall-count distractors (0, 1, 2/3 different, same wall).
DISTRACTOR_POOL_FACTS.append(OptionFact("hits_0_walls"))
DISTRACTOR_POOL_FACTS.append(OptionFact("hits_1_wall"))
for n in (2, 3):
    DISTRACTOR_POOL_FACTS.append(OptionFact("hits_n_diff_walls", (n,)))
for n in (2, 3):
    DISTRACTOR_POOL_FACTS.append(OptionFact("hits_same_wall_n_times", (n,)))

# Sequence-of-wall distractors (first / second / third wall hit by label).
for order_kind, idx in (
    ("first_wall_hit", 0),
    ("second_wall_hit", 1),
    ("third_wall_hit", 2),
):
    for name in WALL_NAMES:
        DISTRACTOR_POOL_FACTS.append(OptionFact(order_kind, (name,)))


def sample_multilabel_from_facts(
    true_facts: Sequence[OptionFact],
    pool_facts: Sequence[OptionFact],
    total: int,
    num_correct: int,
    tense: Tense,
    renderer: OptionRenderer,
) -> Tuple[List[str], List[int]]:
    """
    Sample a mixture of true and distractor options, returning rendered strings
    and the indices of the correct options.
    """
    if not true_facts:
        # Fallback: if we somehow have no information, default to "not pocketed".
        true_facts = [OptionFact("not_pocketed")]

    num_correct = max(1, min(num_correct, len(true_facts), total))

    # Choose which facts will be marked as correct.
    chosen_true: List[OptionFact] = random_sample(true_facts, num_correct)

    # Filter distractors for logical consistency and to avoid duplicates.
    chosen_set = set(chosen_true)
    distractor_candidates = [f for f in pool_facts if f not in chosen_set]
    num_distractors_needed = max(0, total - num_correct)
    distractors: List[OptionFact] = random_sample(
        distractor_candidates, min(num_distractors_needed, len(distractor_candidates))
    )

    labeled: List[Tuple[OptionFact, bool]] = [
        (f, True) for f in chosen_true
    ] + [(f, False) for f in distractors]

    # Remove duplicate facts while preserving order, then shuffle while tracking
    # which entries are correct.
    deduped: List[Tuple[OptionFact, bool]] = []
    seen_facts = set()
    for fact, is_correct in labeled:
        if fact in seen_facts:
            continue
        seen_facts.add(fact)
        deduped.append((fact, is_correct))

    import random

    random.shuffle(deduped)

    options = [renderer.render(fact, tense) for fact, _ in deduped]
    ground_truth = [idx for idx, (_, is_correct) in enumerate(deduped) if is_correct]
    return options, ground_truth


def random_sample(items: Sequence[OptionFact], k: int) -> List[OptionFact]:
    """Helper that gracefully handles ``k > len(items)``."""
    import random

    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    return random.sample(list(items), k)
