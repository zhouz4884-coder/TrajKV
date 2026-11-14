"""Backend-independent precision policies and explicit read-cost accounting.

No tensors are changed here. The executor owns trial/replay buffers and supplies
the byte counts; projected costs must not be reported as measured GPU traffic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import math
import random
from typing import Any, Iterable, Mapping


class Availability(str, Enum):
    ONLINE = "online"
    REFERENCE = "reference"
    FUTURE = "future"


class FeatureAvailabilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Feature:
    value: Any
    availability: Availability = Availability.ONLINE
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "availability", Availability(self.availability))

    def online_value(self) -> Any:
        if self.availability != Availability.ONLINE:
            raise FeatureAvailabilityError(
                f"Feature from {self.source or 'unspecified source'} is "
                f"{self.availability.value}, not online"
            )
        return self.value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """Half-open absolute token interval, including one-token short answers."""

    start: int
    end: int
    label: str = "fact"
    availability: Availability = Availability.ONLINE
    source: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("TokenSpan needs 0 <= start < end")
        object.__setattr__(self, "availability", Availability(self.availability))

    def overlaps(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end


@dataclass(frozen=True, slots=True)
class Cost:
    """Reads above one base pass are residual + probe + replay reads.

    probe_read_bytes means EXTRA feature-extraction traffic. The normal low
    precision trial is the base pass. replay_read_bytes is reread base traffic
    during a refined replay; its residual traffic is counted exactly once in
    residual_read_bytes. Compute times follow the same disjoint convention.
    """

    base_read_bytes: int = 0
    residual_read_bytes: int = 0
    probe_read_bytes: int = 0
    replay_read_bytes: int = 0
    base_compute_us: float = 0.0
    probe_compute_us: float = 0.0
    replay_compute_us: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name.endswith("_bytes"):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{item.name} must be a nonnegative integer")
            elif not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} must be finite and nonnegative")

    @property
    def extra_read_bytes(self) -> int:
        return self.residual_read_bytes + self.probe_read_bytes + self.replay_read_bytes

    @property
    def total_read_bytes(self) -> int:
        return self.base_read_bytes + self.extra_read_bytes

    @property
    def total_compute_us(self) -> float:
        return self.base_compute_us + self.probe_compute_us + self.replay_compute_us

    def __add__(self, other: Cost) -> Cost:
        if not isinstance(other, Cost):
            return NotImplemented
        return Cost(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def to_dict(self) -> dict[str, int | float]:
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        result.update(extra_read_bytes=self.extra_read_bytes,
                      total_read_bytes=self.total_read_bytes,
                      total_compute_us=self.total_compute_us)
        return result


@dataclass(frozen=True, slots=True)
class Budget:
    max_extra_read_bytes: int | None = None
    max_total_read_bytes: int | None = None
    max_probe_read_bytes: int | None = None
    max_replay_read_bytes: int | None = None
    max_compute_us: float | None = None
    max_refined_events: int | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None:
                continue
            if item.name == "max_compute_us":
                if not math.isfinite(value) or value < 0:
                    raise ValueError("max_compute_us must be finite and nonnegative")
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{item.name} must be a nonnegative integer or None")

    def allows(self, cost: Cost, refined_events: int = 0) -> bool:
        checks = ((self.max_extra_read_bytes, cost.extra_read_bytes),
                  (self.max_total_read_bytes, cost.total_read_bytes),
                  (self.max_probe_read_bytes, cost.probe_read_bytes),
                  (self.max_replay_read_bytes, cost.replay_read_bytes),
                  (self.max_compute_us, cost.total_compute_us),
                  (self.max_refined_events, refined_events))
        return all(limit is None or value <= limit for limit, value in checks)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadEvent:
    event_id: str
    session_id: str
    turn: int
    token_start: int
    token_end: int
    base_read_bytes: int
    residual_read_bytes: int
    step_index: int = 0
    phase: str = "decode"
    layer_group: str = "all"
    head_group: str = "all"
    unit_key: str = ""
    features: Mapping[str, Feature] = field(default_factory=dict)
    carrier_spans: tuple[TokenSpan, ...] = ()
    probe_read_bytes: int = 0
    replay_read_bytes: int | None = None
    base_compute_us: float = 0.0
    probe_compute_us: float = 0.0
    replay_compute_us: float = 0.0
    cost_source: str = "declared"

    def __post_init__(self) -> None:
        if not self.event_id or not self.session_id:
            raise ValueError("event_id and session_id must be nonempty")
        if self.turn < 0 or self.step_index < 0:
            raise ValueError("turn and step_index must be nonnegative")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("ReadEvent needs 0 <= token_start < token_end")
        if any(not isinstance(v, Feature) for v in self.features.values()):
            raise TypeError("Every event feature must carry an explicit Feature availability label")
        object.__setattr__(self, "carrier_spans", tuple(self.carrier_spans))
        self.cost(refined=True, two_pass=True, probed=True)  # validate supplied costs

    @property
    def key(self) -> str:
        return self.unit_key or f"{self.phase}|{self.layer_group}|{self.head_group}"

    def online_feature(self, name: str) -> Any:
        return self.features[name].online_value()

    def margin(self) -> float:
        result = _number(self.online_feature("margin"), "margin")
        if result < 0:
            raise ValueError("top1-top2 margin cannot be negative")
        return result

    def is_carrier(self) -> bool:
        # A label is structural evidence supplied by the caller, never inferred
        # here from length. One-token yes/no, counts and attributes are allowed.
        flagged = False
        if "fact_carrier" in self.features:
            value = self.online_feature("fact_carrier")
            if not isinstance(value, bool):
                raise ValueError("fact_carrier must be a bool Feature")
            flagged = value
        for span in self.carrier_spans:
            if span.availability != Availability.ONLINE:
                raise FeatureAvailabilityError("Carrier spans used online must be online-labelled")
            flagged = flagged or span.overlaps(self.token_start, self.token_end)
        return flagged

    def cost(self, *, refined: bool, two_pass: bool = False, probed: bool = False) -> Cost:
        return Cost(
            base_read_bytes=self.base_read_bytes,
            residual_read_bytes=self.residual_read_bytes if refined else 0,
            probe_read_bytes=self.probe_read_bytes if probed else 0,
            replay_read_bytes=(self.base_read_bytes if self.replay_read_bytes is None
                               else self.replay_read_bytes) if refined and two_pass else 0,
            base_compute_us=self.base_compute_us,
            probe_compute_us=self.probe_compute_us if probed else 0.0,
            replay_compute_us=self.replay_compute_us if refined and two_pass else 0.0)


@dataclass(frozen=True, slots=True)
class Decision:
    event_id: str
    refined: bool
    policy: str
    score: float | None
    reason: str
    projected_cost: Cost
    oracle: bool = False


@dataclass(frozen=True, slots=True)
class Allocation:
    decisions: tuple[Decision, ...]
    projected_cost: Cost
    total_projected_cost: Cost
    budget_ok: bool
    oracle: bool = False

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return tuple(d.event_id for d in self.decisions if d.refined)


class Policy:
    """Online policies inspect events in supplied causal order, never rank
    future events by their scores. Keep one instance to preserve schedule/RNG
    state across small allocation calls. Allocation is a plan, not execution.
    """

    name = "policy"
    two_pass = False
    probed = False

    def propose(self, event: ReadEvent) -> tuple[bool, float | None, str]:
        raise NotImplementedError

    def allocate(self, events: Iterable[ReadEvent], budget: Budget, *,
                 starting_cost: Cost = Cost(), starting_refined_events: int = 0) -> Allocation:
        cost = Cost()
        count = starting_refined_events
        decisions = []
        seen = set()
        for event in events:
            if event.event_id in seen:
                raise ValueError(f"Duplicate event_id: {event.event_id}")
            seen.add(event.event_id)
            proposed, score, reason = self.propose(event)
            high = event.cost(refined=True, two_pass=self.two_pass, probed=self.probed)
            refined = proposed and budget.allows(starting_cost + cost + high, count + 1)
            if proposed and not refined:
                reason = "budget_exhausted"
            chosen = high if refined else event.cost(refined=False, probed=self.probed)
            cost = cost + chosen
            count += int(refined)
            decisions.append(Decision(event.event_id, refined, self.name, score, reason, chosen))
        total = starting_cost + cost
        return Allocation(tuple(decisions), cost, total, budget.allows(total, count))


class UniformPolicy(Policy):
    """A causal evenly-spaced schedule; rate=1 attempts every event."""

    name = "uniform"

    def __init__(self, rate: float = 1.0) -> None:
        if not 0 <= rate <= 1:
            raise ValueError("rate must lie in [0,1]")
        self.rate = rate
        self._credit = 0.0

    def propose(self, event: ReadEvent) -> tuple[bool, float, str]:
        self._credit += self.rate
        selected = self._credit >= 1.0 - 1e-12
        if selected:
            self._credit -= 1.0
        return selected, self.rate, "uniform_schedule"


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, rate: float = 0.5, seed: int = 0) -> None:
        if not 0 <= rate <= 1:
            raise ValueError("rate must lie in [0,1]")
        self.rate = rate
        self._rng = random.Random(seed)

    def propose(self, event: ReadEvent) -> tuple[bool, float, str]:
        draw = self._rng.random()
        return draw < self.rate, draw, "seeded_random"


class PeriodicPolicy(Policy):
    name = "periodic"

    def __init__(self, period: int = 4, width: int = 1, offset: int = 0) -> None:
        if period <= 0 or not 0 <= width <= period:
            raise ValueError("Require period > 0 and 0 <= width <= period")
        self.period, self.width, self.offset = period, width, offset

    def propose(self, event: ReadEvent) -> tuple[bool, float, str]:
        selected = (event.step_index - self.offset) % self.period < self.width
        return selected, float(selected), "periodic_window"


class CurrentMarginPolicy(Policy):
    name = "current_margin"
    two_pass = True
    probed = True

    def __init__(self, threshold: float) -> None:
        self.threshold = _number(threshold, "threshold")
        if self.threshold < 0:
            raise ValueError("margin threshold must be nonnegative")

    def propose(self, event: ReadEvent) -> tuple[bool, float, str]:
        margin = event.margin()
        return margin <= self.threshold, margin, "low_margin" if margin <= self.threshold else "high_margin"


class FactCarrierMarginPolicy(CurrentMarginPolicy):
    name = "fact_carrier_margin"

    def propose(self, event: ReadEvent) -> tuple[bool, float, str]:
        proposed, margin, reason = super().propose(event)
        carrier = event.is_carrier()
        return proposed and carrier, margin, reason if carrier else "not_carrier"


class FutureGainTablePolicy(Policy):
    name = "future_gain_table"

    def __init__(self, table: Mapping[str, float | Feature], *,
                 min_gain_per_byte: float = 0.0, calibration_id: str = "") -> None:
        self.table = dict(table)
        self.min_gain_per_byte = _number(min_gain_per_byte, "min_gain_per_byte")
        self.calibration_id = calibration_id

    def propose(self, event: ReadEvent) -> tuple[bool, float, str]:
        value = self.table.get(event.key, 0.0)
        gain = _number(value.online_value() if isinstance(value, Feature) else value, "predicted_gain")
        score = gain / max(1, event.residual_read_bytes)
        return gain > 0 and score >= self.min_gain_per_byte, score, "calibrated_future_gain"


class OfflineOraclePolicy(Policy):
    """Greedy future-gain/extra-byte upper reference, NOT an exact optimizer."""

    name = "offline_oracle"

    def __init__(self, feature: str = "oracle_future_gain", *, two_pass: bool = False) -> None:
        self.feature = feature
        self.two_pass = two_pass

    def allocate(self, events: Iterable[ReadEvent], budget: Budget, *,
                 starting_cost: Cost = Cost(), starting_refined_events: int = 0) -> Allocation:
        events = tuple(events)
        if len({e.event_id for e in events}) != len(events):
            raise ValueError("Duplicate event_id")
        gains = []
        low = Cost()
        for event in events:
            feature = event.features[self.feature]
            if feature.availability != Availability.FUTURE:
                raise FeatureAvailabilityError("Oracle gains must be explicitly labelled future")
            gain = _number(feature.value, self.feature)
            high = event.cost(refined=True, two_pass=self.two_pass)
            extra = high.extra_read_bytes
            gains.append((gain / max(1, extra), gain))
            low = low + event.cost(refined=False)
        selected = set()
        cost = low
        count = starting_refined_events
        for index in sorted(range(len(events)), key=lambda i: (-gains[i][0], i)):
            if gains[index][1] <= 0:
                continue
            event = events[index]
            increment = Cost(residual_read_bytes=event.residual_read_bytes,
                             replay_read_bytes=(event.base_read_bytes if event.replay_read_bytes is None
                                                else event.replay_read_bytes) if self.two_pass else 0,
                             replay_compute_us=event.replay_compute_us if self.two_pass else 0.0)
            if budget.allows(starting_cost + cost + increment, count + 1):
                selected.add(index)
                cost = cost + increment
                count += 1
        decisions = tuple(Decision(e.event_id, i in selected, self.name, gains[i][0],
                                   "future_greedy_selected" if i in selected else "not_selected",
                                   e.cost(refined=i in selected, two_pass=self.two_pass), True)
                          for i, e in enumerate(events))
        total = starting_cost + cost
        return Allocation(decisions, cost, total, budget.allows(total, count), True)


def make_policy(name: str, **kwargs: Any) -> Policy:
    classes = {cls.name: cls for cls in (UniformPolicy, RandomPolicy, PeriodicPolicy,
               CurrentMarginPolicy, FactCarrierMarginPolicy, FutureGainTablePolicy, OfflineOraclePolicy)}
    try:
        cls = classes[name]
    except KeyError as exc:
        raise ValueError(f"Unknown policy {name!r}; expected one of {tuple(classes)}") from exc
    return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class CommitRecord:
    event_id: str
    refined: bool
    cost: Cost
    cost_source: str
    budget_ok: bool
    reason: str


class CostLedger:
    """Records consumed work, including overruns; never hides spent cost."""

    def __init__(self) -> None:
        self.total = Cost()
        self.refined_events = 0
        self.records: list[CommitRecord] = []
        self._seen: set[str] = set()

    def record(self, event_id: str, refined: bool, cost: Cost, budget: Budget, *,
               cost_source: str, reason: str = "") -> CommitRecord:
        if event_id in self._seen:
            raise ValueError(f"Event already committed: {event_id}")
        self.total = self.total + cost
        self.refined_events += int(refined)
        record = CommitRecord(event_id, refined, cost, cost_source,
                              budget.allows(self.total, self.refined_events), reason)
        self.records.append(record)
        self._seen.add(event_id)
        return record


class MarginTransaction:
    """A logical trial/replay/commit state machine. It does not mutate KV.

    Executor must keep trial KV uncommitted, restore the pre-trial checkpoint
    before replay, then commit exactly the selected branch at all changed layers.
    """

    def __init__(self, event: ReadEvent, policy: CurrentMarginPolicy,
                 budget: Budget, ledger: CostLedger) -> None:
        if not isinstance(policy, CurrentMarginPolicy):
            raise TypeError("MarginTransaction requires a margin policy")
        self.event, self.policy, self.budget, self.ledger = event, policy, budget, ledger
        self.state = "new"
        self.decision: Decision | None = None
        self._trial_cost = Cost()

    def trial(self, *, actual_trial_cost: Cost | None = None) -> Decision:
        if self.state != "new":
            raise RuntimeError("trial can run exactly once")
        self._trial_cost = actual_trial_cost or self.event.cost(refined=False, probed=True)
        proposed, score, reason = self.policy.propose(self.event)
        replay = self._replay_cost()
        refined = proposed and self.budget.allows(
            self.ledger.total + self._trial_cost + replay, self.ledger.refined_events + 1)
        if proposed and not refined:
            reason = "budget_exhausted"
        self.decision = Decision(self.event.event_id, refined, self.policy.name, score, reason,
                                 self._trial_cost + replay if refined else self._trial_cost)
        self.state = "trial"
        return self.decision

    def _replay_cost(self) -> Cost:
        return Cost(residual_read_bytes=self.event.residual_read_bytes,
                    replay_read_bytes=self.event.base_read_bytes if self.event.replay_read_bytes is None
                                      else self.event.replay_read_bytes,
                    replay_compute_us=self.event.replay_compute_us)

    def commit(self, *, refined: bool, actual_replay_cost: Cost | None = None,
               cost_source: str | None = None) -> CommitRecord:
        if self.state != "trial" or self.decision is None:
            raise RuntimeError("commit requires one uncommitted trial")
        if refined and not self.decision.refined:
            raise RuntimeError("Refinement was not approved by trial/budget")
        if not refined and actual_replay_cost is not None:
            raise ValueError("A non-refined commit cannot include refined replay cost")
        cost = self._trial_cost + ((actual_replay_cost or self._replay_cost()) if refined else Cost())
        record = self.ledger.record(self.event.event_id, refined, cost, self.budget,
                    cost_source=cost_source or self.event.cost_source, reason=self.decision.reason)
        self.state = "committed"
        return record

    def abort(self, *, cost_source: str | None = None) -> CommitRecord:
        """Close without refinement; a consumed trial still costs resources."""
        record = self.commit(refined=False, cost_source=cost_source)
        return record
