"""Rule fault sequence pairing (read-only cause→effect matching).

Given the **already stored** fault intervals of two different rules — an
antecedent (前因) rule and a subsequent (后续) rule — the pairing links each
antecedent fault to a later subsequent fault, anchoring on interval **start**
timestamps:

* only faults whose **start falls inside** the half-open UTC query window
  ``[start, end)`` participate (the caller filters; interval ends and any
  overlap with the window do not matter);
* a pair ``(a, b)`` is admissible when the subsequent fault starts **strictly
  later** than the antecedent fault and no more than ``max_lag`` seconds after
  it: ``a.start < b.start <= a.start + max_lag``;
* every fault interval takes part in **at most one** pair, and the matching
  **preserves time order** (it is non-crossing: pairing ``a₁→b₂`` and
  ``a₂→b₁`` with ``a₁`` before ``a₂`` is not allowed) — cause and effect stay
  consistently ordered;
* among all such one-to-one matchings the chosen one first **maximizes the
  number of pairs**, then **minimizes the total lag** (sum of
  ``b.start − a.start`` over the pairs);
* remaining ties are resolved **stably by fault id**: both sides are ordered
  by ``(start, id)`` and, among equally optimal matchings, the one whose
  paired ``(antecedent_id, subsequent_id)`` sequence is lexicographically
  smallest wins, so the same stored faults always yield the same pairing.

The optimization is an LCS-style suffix dynamic program over the two
start-ordered fault lists.  Besides the scalar ``(pair_count, total_lag)``
objective each cell also remembers the first pair of its optimal solution;
because two candidate moves whose first pairs coincide always continue
identically, comparing first pairs is enough to make the reconstruction
lexicographically exact without storing whole sequences per cell.

Everything here is a pure function of the supplied fault outlines: nothing is
read from or written to any table, and rule evaluation is never triggered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_EPS = 1e-7  # seconds of slack for boundary comparisons (float timestamps)
# Safety valve: the DP is O(len(antecedent) × len(subsequent)) in time and
# memory, so absurdly large fault lists are refused instead of hanging.
MAX_CELLS = 1_000_000


@dataclass(frozen=True)
class FaultAnchor:
    """A stored fault interval reduced to its pairing-relevant outline."""

    id: int
    start: float
    end: float

    def to_dict(self) -> dict:
        return {"fault_id": self.id, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class FaultPair:
    """One antecedent → subsequent link with its start-to-start lag."""

    antecedent: FaultAnchor
    subsequent: FaultAnchor

    @property
    def lag_seconds(self) -> float:
        return self.subsequent.start - self.antecedent.start

    def to_dict(self) -> dict:
        return {
            "antecedent_fault_id": self.antecedent.id,
            "antecedent_start": self.antecedent.start,
            "antecedent_end": self.antecedent.end,
            "subsequent_fault_id": self.subsequent.id,
            "subsequent_start": self.subsequent.start,
            "subsequent_end": self.subsequent.end,
            "lag_seconds": self.lag_seconds,
        }


@dataclass(frozen=True)
class PairingResult:
    """Pairs plus both sides' unmatched faults and the pairing rates."""

    max_lag_seconds: float
    pairs: list[FaultPair] = field(default_factory=list)
    unmatched_antecedent: list[FaultAnchor] = field(default_factory=list)
    unmatched_subsequent: list[FaultAnchor] = field(default_factory=list)
    total_antecedent: int = 0
    total_subsequent: int = 0

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    @property
    def total_lag_seconds(self) -> float:
        return sum(p.lag_seconds for p in self.pairs)

    def to_dict(self) -> dict:
        matched_a = self.pair_count
        matched_b = self.pair_count
        return {
            "max_lag_seconds": self.max_lag_seconds,
            "pair_count": self.pair_count,
            "total_lag_seconds": self.total_lag_seconds,
            "pairs": [p.to_dict() for p in self.pairs],
            "antecedent": {
                "total": self.total_antecedent,
                "matched": matched_a,
                "unmatched": self.total_antecedent - matched_a,
                "match_rate": (
                    matched_a / self.total_antecedent if self.total_antecedent else 0.0
                ),
                "unmatched_faults": [f.to_dict() for f in self.unmatched_antecedent],
            },
            "subsequent": {
                "total": self.total_subsequent,
                "matched": matched_b,
                "unmatched": self.total_subsequent - matched_b,
                "match_rate": (
                    matched_b / self.total_subsequent if self.total_subsequent else 0.0
                ),
                "unmatched_faults": [f.to_dict() for f in self.unmatched_subsequent],
            },
        }


def _better(count_x: int, lag_x: float, first_x, count_y: int, lag_y: float, first_y) -> bool:
    """Whether objective ``(count_x, lag_x, first_x)`` beats ``(..._y)``.

    Lexicographic: more pairs always win; on equal pair count the smaller
    total lag wins; on equal lag the lexicographically smaller first
    ``(antecedent_id, subsequent_id)`` pair wins.  Two moves with the same
    first pair always continue identically, so comparing first pairs is
    enough.  ``first`` is ``None`` exactly when the count is zero.
    """
    if count_x != count_y:
        return count_x > count_y
    if abs(lag_x - lag_y) > _EPS:
        return lag_x < lag_y
    if first_x is None or first_y is None:
        return False
    return first_x < first_y


def pair_faults(antecedent, subsequent, max_lag_seconds: float) -> PairingResult:
    """Optimally pair antecedent faults with later subsequent faults.

    ``antecedent`` / ``subsequent`` are iterables of :class:`FaultAnchor`
    (already filtered to the query window by the caller); they are sorted by
    ``(start, id)`` here.  ``max_lag_seconds`` must be a positive finite
    number of seconds.  See the module docstring for the admissibility,
    optimality and tie-breaking rules.
    """
    max_lag = float(max_lag_seconds)
    if not math.isfinite(max_lag) or max_lag <= 0:
        raise ValueError("max_lag_seconds must be a positive finite number of seconds")

    a_list = sorted(antecedent, key=lambda f: (f.start, f.id))
    b_list = sorted(subsequent, key=lambda f: (f.start, f.id))
    n, m = len(a_list), len(b_list)
    if n * m > MAX_CELLS:
        raise ValueError(
            f"too many faults to pair ({n} × {m}); narrow the query range"
        )

    # Suffix DP: best objective for matching a_list[i:] with b_list[j:].
    # count/lag hold the scalar objective; first holds the first
    # (antecedent_id, subsequent_id) pair of the optimal solution (None when
    # no pair is possible), which makes the id tie-break exact.
    count = [[0] * (m + 1) for _ in range(n + 1)]
    lag = [[0.0] * (m + 1) for _ in range(n + 1)]
    first: list[list[tuple[int, int] | None]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]

    for i in range(n - 1, -1, -1):
        a = a_list[i]
        for j in range(m - 1, -1, -1):
            # Move 1: leave a_list[i] unpaired.
            best_c, best_l, best_f = count[i + 1][j], lag[i + 1][j], first[i + 1][j]
            # Move 2: leave b_list[j] unpaired.
            if _better(
                count[i][j + 1], lag[i][j + 1], first[i][j + 1],
                best_c, best_l, best_f,
            ):
                best_c, best_l, best_f = count[i][j + 1], lag[i][j + 1], first[i][j + 1]
            # Move 3: pair a_list[i] with b_list[j] when admissible.
            gap = b_list[j].start - a.start
            if _EPS < gap <= max_lag + _EPS:
                c3 = count[i + 1][j + 1] + 1
                l3 = lag[i + 1][j + 1] + gap
                f3 = (a.id, b_list[j].id)
                if _better(c3, l3, f3, best_c, best_l, best_f):
                    best_c, best_l, best_f = c3, l3, f3
            count[i][j], lag[i][j], first[i][j] = best_c, best_l, best_f

    # Reconstruct the lexicographically exact optimal matching: at each cell
    # take the move whose first pair equals the cell's recorded first pair.
    pairs: list[FaultPair] = []
    paired_a: set[int] = set()
    paired_b: set[int] = set()
    i = j = 0
    while i < n and j < m and count[i][j] > 0:
        a, b = a_list[i], b_list[j]
        gap = b.start - a.start
        if _EPS < gap <= max_lag + _EPS and first[i][j] == (a.id, b.id):
            pairs.append(FaultPair(antecedent=a, subsequent=b))
            paired_a.add(a.id)
            paired_b.add(b.id)
            i += 1
            j += 1
        elif first[i][j] == first[i + 1][j]:
            i += 1
        else:
            j += 1

    return PairingResult(
        max_lag_seconds=max_lag,
        pairs=pairs,
        unmatched_antecedent=[a for a in a_list if a.id not in paired_a],
        unmatched_subsequent=[b for b in b_list if b.id not in paired_b],
        total_antecedent=n,
        total_subsequent=m,
    )
