"""All model time coordinates are timezone-aware UTC datetimes."""
from datetime import datetime, timedelta, timezone

UTC = timezone.utc


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamp needs timezone, e.g. 2026-10-20T18:00:00Z")
    return dt.astimezone(UTC)


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("Refusing naive timestamp")
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def utc_hours(start: datetime, end: datetime):
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Use timezone-aware UTC start/end")
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        yield cur
        cur += timedelta(hours=1)
