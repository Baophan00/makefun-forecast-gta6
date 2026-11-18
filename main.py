"""make.fun gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

One hour into GTA VI's release day we know hour 1's print p. For every past
launch we find its own release-day hour 1 and hours 2..24, take log ratios
log(v_h / v_1), pool those ratios across launches, read the five score levels
off each hour's pooled distribution, blend the result toward a simple prior
(the house persistence shape) when the sample is small, rescale by p, and
emit rounded, finite, positive, monotone quantiles.

No network, no randomness, no clock reads: iteration is over sorted file names
and every arithmetic step is deterministic. Runs in milliseconds.
"""

import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Score levels, in the order each row's q array lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at those levels (the house prior's log-space shape).
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
# Log-space spread per sqrt(hour) of the prior.
SIGMA = 0.15
# Pseudo-count: weight on the prior is PSEUDO / (n + PSEUDO) per hour.
PSEUDO = 2.0
# Hard ceiling on a viewer count.
MAX_VIEWERS = 1e9
HOURS = range(2, 25)
# A comparable's own hour-to-hour move is capped to this multiplicative band.
MIN_RATIO = 0.02
MAX_RATIO = 40.0
# Below this many usable comparables, nudge the forecast toward the global curve.
MIN_COMPARABLES = 5

# Keys under which a comparable may keep its release-day hourly points.
_HOURLY_KEYS = (
    "hourly", "hourly_points", "hours", "points", "history", "history_hourly",
    "data", "series", "values", "readings", "viewers", "ccu", "series_hourly",
)
# Keys under which a point may carry its viewers/CCU.
_VIEWER_KEYS = (
    "viewers", "ccu", "viewers_mean", "mean_viewers", "value", "v",
    "watching", "peak_viewers", "count",
)
# Keys under which a comparable may carry its release-day start.
_DAY_KEYS = ("day_start", "start_date", "release_day", "date", "day", "start")
# Keys under which a point may carry a timestamp.
_TIME_KEYS = ("from", "until", "time", "timestamp", "ts", "start", "at", "t")


def _num(x):
    """Finite float, or None."""
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _dt(x):
    """Timestamp -> naive UTC datetime, or None."""
    if isinstance(x, datetime):
        d = x
    elif isinstance(x, date):
        d = datetime(x.year, x.month, x.day)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        v = _num(x)
        if v is None:
            return None
        try:
            d = datetime.fromtimestamp(v, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                d = datetime.combine(date.fromisoformat(s[:10]), datetime.min.time())
            except ValueError:
                return None
    else:
        return None
    if d.tzinfo is not None:
        d = d.astimezone(timezone.utc).replace(tzinfo=None)
    return d


def _day_start(comp):
    if not isinstance(comp, dict):
        return None
    for key in _DAY_KEYS:
        if key in comp:
            d = _dt(comp[key])
            if d is not None:
                return d
    return None


def _points(comp):
    """The comparable's list of hourly points, however it is stored."""
    if isinstance(comp, list):
        return comp if comp else None
    if not isinstance(comp, dict):
        return None
    for key in _HOURLY_KEYS:
        value = comp.get(key)
        if isinstance(value, list) and value:
            return value
        if isinstance(value, dict):
            pairs = []
            for hour, viewers in value.items():
                h, v = _num(hour), _num(viewers)
                if h is not None and v is not None:
                    pairs.append({"hour": int(h), "viewers": v})
            if pairs:
                return pairs
    return None


def _point_stamp(point):
    """Point timestamp, from its time fields or its own date plus hour."""
    if not isinstance(point, dict):
        return None
    for key in _TIME_KEYS:
        if key in point:
            stamp = _dt(point[key])
            if stamp is not None:
                return stamp
    for key in ("date", "day", "day_start"):
        if key in point:
            day = _dt(point[key])
            if day is not None:
                hour = None
                for hkey in ("hour", "h", "release_hour"):
                    value = _num(point.get(hkey))
                    if value is not None and 1 <= value <= 24:
                        hour = int(value)
                        break
                if hour is not None:
                    return day.replace(hour=hour - 1, minute=0, second=0, microsecond=0)
                return day
    return None


def _point_hour(point, day0):
    """Hour 1..24 counted from the comparable's own day_start, else None."""
    stamp = _point_stamp(point)
    if stamp is not None:
        if day0 is not None:
            delta = (stamp - day0).total_seconds() / 3600.0
            hour = math.floor(delta) + 1
            return hour if 0 <= delta < 24 and 1 <= hour <= 24 else None
        hour = stamp.hour + 1
        return hour if 1 <= hour <= 24 else None
    if isinstance(point, dict):
        for key in ("hour", "h", "release_hour", "hour_index"):
            value = _num(point.get(key))
            if value is not None and 1 <= value <= 24:
                return int(value)
    elif isinstance(point, (list, tuple)) and point:
        value = _num(point[0])
        if value is not None and 1 <= value <= 24:
            return int(value)
    return None


def _point_viewers(point):
    if isinstance(point, dict):
        for key in _VIEWER_KEYS:
            if key in point:
                v = _num(point[key])
                if v is not None and v > 0:
                    return v
        reads = point.get("reads")
        if isinstance(reads, list):
            vals = []
            for read in reads:
                v = _num(read) if not isinstance(read, dict) else _point_viewers(read)
                if v is not None and v > 0:
                    vals.append(v)
            if vals:
                return sum(vals) / len(vals)
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        v = _num(point[1])
        if v is not None and v > 0:
            return v
    return None


def _log_ratios(comp):
    """Hour -> log(v_hour / v_hour1) for one comparable's release day."""
    points = _points(comp)
    if not points:
        return {}
    day0 = _day_start(comp)
    totals, counts = {}, {}
    for point in points:
        hour = _point_hour(point, day0)
        if hour is None:
            continue
        viewers = _point_viewers(point)
        if viewers is None or viewers <= 0:
            continue
        totals[hour] = totals.get(hour, 0.0) + viewers
        counts[hour] = counts.get(hour, 0) + 1
    if 1 not in totals or counts[1] == 0:
        return {}
    base = totals[1] / counts[1]
    if not math.isfinite(base) or base <= 0:  # a zero hour 1 yields no ratios
        return {}
    out = {}
    for hour in HOURS:
        if hour in totals and counts[hour] > 0:
            ratio = (totals[hour] / counts[hour]) / base
            if not math.isfinite(ratio) or ratio <= 0:
                continue
            ratio = min(max(ratio, MIN_RATIO), MAX_RATIO)
            log_ratio = math.log(ratio)
            if math.isfinite(log_ratio):
                out[hour] = log_ratio
    return out


def _unlock(hour):
    """Global launch-day shape in log space: rise to a midday peak, ease down late."""
    u = (hour - 1) / 23.0
    return 0.30 * (4.0 * u * (1.0 - u)) ** 0.85 + 0.04 * u


def _quantile(sorted_values, tau):
    """Linear-interpolation (type-7) quantile of an already sorted sample."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


def _round2(x):
    """Round half up to two decimals (positive x), int when whole."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def forecast_case(case):
    known = case.get("known") or []
    print1 = None
    for row in known:
        if isinstance(row, dict) and _num(row.get("hour")) == 1:
            print1 = _num(row.get("viewers"))
            break
    if print1 is None and known and isinstance(known[0], dict):
        print1 = _num(known[0].get("viewers"))
    if print1 is None or print1 <= 0:
        print1 = 1.0
    p = max(print1, 1.0)

    pooled = {hour: [] for hour in HOURS}
    comparables = case.get("comparables") or []
    usable = 0
    if isinstance(comparables, list):
        for comp in comparables:
            ratios = _log_ratios(comp)
            if ratios:
                usable += 1
                for hour, ratio in ratios.items():
                    pooled[hour].append(ratio)
    fallback = max(0.0, (MIN_COMPARABLES - usable) / MIN_COMPARABLES)

    rows = []
    for hour in HOURS:
        sample = pooled[hour]
        sample.sort()
        n = len(sample)
        weight = n / (n + PSEUDO) if n else 0.0
        root = math.sqrt(hour - 1)
        quantiles = []
        for tau, z in zip(LEVELS, Z):
            prior = z * SIGMA * root
            if n:
                log_ratio = weight * _quantile(sample, tau) + (1.0 - weight) * prior
            else:
                log_ratio = prior
            if fallback:
                log_ratio += fallback * _unlock(hour)
            value = p * math.exp(log_ratio)
            if not math.isfinite(value) or value <= 0:
                value = p
            quantiles.append(_round2(min(value, MAX_VIEWERS)))
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        if quantiles[0] <= 0:
            quantiles[0] = 0.01
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
