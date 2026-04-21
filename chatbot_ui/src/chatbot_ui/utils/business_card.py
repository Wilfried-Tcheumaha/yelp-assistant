"""Yelp-style rating stars and business card HTML for Streamlit."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

YELP_STARS_CSS = """
<style>
/* -- Rating stars -- */
.yelp-rating { display: inline-flex; gap: 2px; vertical-align: middle; }
.yelp-rating .yelp-star {
    width: 22px; height: 22px;
    display: inline-flex; align-items: center; justify-content: center;
    background: #dcdcdc; color: #fff;
    border-radius: 4px; font-size: 13px; line-height: 1;
    position: relative; overflow: hidden;
}
.yelp-rating .yelp-star.filled { background: #ef6c00; }
.yelp-rating .yelp-star.half::before {
    content: ""; position: absolute; top: 0; left: 0;
    width: 50%; height: 100%; background: #ef6c00;
}
.yelp-rating .yelp-star span { position: relative; z-index: 1; }

/* -- Business card -- */
.yelp-card { margin-bottom: 4px; }
.yelp-card__name {
    font-weight: 800; font-size: 1.1rem; color: #111;
    line-height: 1.2; margin-bottom: 6px;
}
.yelp-card__rating-line {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 6px;
}
.yelp-card__score { font-weight: 700; color: #111; font-size: 0.95rem; }
.yelp-card__reviews { color: #757575; font-size: 0.9rem; }
.yelp-card__meta {
    color: #333; font-size: 0.9rem; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}
.yelp-card__meta .pin { color: #555; }
.yelp-card__address { color: #333; text-decoration: none; }
.yelp-card__address:hover { color: #0073bb; text-decoration: underline; }
.yelp-card__status { font-weight: 700; }
.yelp-card__status--closed { color: #d32323; }
.yelp-card__status--open { color: #2e7d32; }
.yelp-card__status-detail { color: #333; font-weight: 400; }
.yelp-card__tags { display: flex; flex-wrap: wrap; gap: 6px; }
.yelp-tag {
    display: inline-block; padding: 3px 12px;
    border: 1px solid #bfbfbf; border-radius: 999px;
    font-size: 0.82rem; color: #111; background: #fff;
    white-space: nowrap;
}
</style>
"""


def render_stars(rating: float | None) -> str:
    """Render a 0-5 rating as Yelp-style orange star boxes (supports half stars)."""
    if rating is None:
        return '<span style="color:#999">—</span>'
    r = max(0.0, min(5.0, float(rating)))
    full = int(r)
    half = 1 if (r - full) >= 0.5 else 0
    empty = 5 - full - half
    boxes = (
        ['<div class="yelp-star filled"><span>★</span></div>'] * full
        + (['<div class="yelp-star half"><span>★</span></div>'] if half else [])
        + ['<div class="yelp-star"><span>★</span></div>'] * empty
    )
    return f'<div class="yelp-rating">{"".join(boxes)}</div>'


def _escape(s: Any) -> str:
    """Minimal HTML escape for user-ish strings we drop into the card."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def _parse_hhmm(s: str) -> int | None:
    """'11:30' -> 690 (minutes since midnight). Returns None on invalid."""
    try:
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def _parse_hours_for_day(hours: dict, day_name: str) -> tuple[int, int] | None:
    """Return (open_mins, close_mins) for the given day, or None if missing.

    `close_mins` may exceed 1440 for overnight ranges (e.g. "22:0-2:0" -> (1320, 1560)).
    """
    entry = hours.get(day_name) if isinstance(hours, dict) else None
    if not isinstance(entry, str) or "-" not in entry:
        return None
    open_s, close_s = entry.split("-", 1)
    open_m, close_m = _parse_hhmm(open_s), _parse_hhmm(close_s)
    if open_m is None or close_m is None:
        return None
    if close_m <= open_m:
        close_m += 1440
    return (open_m, close_m)


def _fmt_time(minutes: int) -> str:
    """1380 -> '11:00 PM'. Normalizes values past midnight."""
    minutes %= 1440
    h, m = divmod(minutes, 60)
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {suffix}"


def compute_open_status(
    hours: dict | None, now: datetime | None = None
) -> dict[str, Any] | None:
    """Compute live open/closed status from a `{"Monday": "11:30-23:0"}` dict.

    Returns `{"open": bool, "label": "Open"|"Closed", "detail": "until 5:30 PM"}`
    or `None` if hours are missing/unparseable.
    """
    if not hours or not isinstance(hours, dict):
        return None

    now = now or datetime.now()
    today_idx = now.weekday()
    today_name = _DAY_NAMES[today_idx]
    now_minutes = now.hour * 60 + now.minute

    # We may still be inside yesterday's overnight slot (e.g. yesterday 22:00 - today 02:00).
    yesterday_slot = _parse_hours_for_day(hours, _DAY_NAMES[(today_idx - 1) % 7])
    if yesterday_slot:
        _, y_close = yesterday_slot
        if y_close > 1440 and now_minutes < (y_close - 1440):
            return {
                "open": True,
                "label": "Open",
                "detail": f"until {_fmt_time(y_close - 1440)}",
            }

    today_slot = _parse_hours_for_day(hours, today_name)
    if today_slot:
        t_open, t_close = today_slot
        if t_open <= now_minutes < t_close:
            return {
                "open": True,
                "label": "Open",
                "detail": f"until {_fmt_time(t_close)}",
            }
        if now_minutes < t_open:
            return {
                "open": False,
                "label": "Closed",
                "detail": f"until {_fmt_time(t_open)}",
            }

    for offset in range(1, 8):
        idx = (today_idx + offset) % 7
        slot = _parse_hours_for_day(hours, _DAY_NAMES[idx])
        if slot:
            label_day = "tomorrow" if offset == 1 else _DAY_NAMES[idx]
            return {
                "open": False,
                "label": "Closed",
                "detail": f"until {_fmt_time(slot[0])} {label_day}",
            }

    return {"open": False, "label": "Closed", "detail": ""}


def render_business_card(item: dict[str, Any]) -> str:
    """Yelp-style card: name, stars + score + reviews, address, status, pills."""
    raw_name = item.get("name") or ""
    raw_address = item.get("address") or ""
    name = _escape(raw_name or "—")
    stars = item.get("stars")
    reviews = item.get("reviews")
    address = _escape(raw_address)
    categories = item.get("categories") or []
    status = compute_open_status(item.get("hours"))

    score_html = (
        f'<span class="yelp-card__score">{stars:.1f}</span>'
        if isinstance(stars, (int, float))
        else ""
    )
    reviews_html = (
        f'<span class="yelp-card__reviews">({reviews} reviews)</span>'
        if isinstance(reviews, int) and reviews is not None
        else ""
    )

    meta_parts: list[str] = []
    if address:
        find_desc = quote_plus(raw_name.strip())
        find_loc = quote_plus(raw_address.strip())
        yelp_url = (
            f"https://www.yelp.com/search?find_desc={find_desc}&find_loc={find_loc}"
        )
        meta_parts.append(
            f'<span class="pin">📍</span>'
            f'<a class="yelp-card__address" href="{yelp_url}" '
            f'target="_blank" rel="noopener noreferrer">{address}</a>'
        )
    if status:
        cls = "open" if status["open"] else "closed"
        detail = _escape(status.get("detail") or "")
        detail_html = f' <span class="yelp-card__status-detail">{detail}</span>' if detail else ""
        meta_parts.append(
            f'<span class="yelp-card__status yelp-card__status--{cls}">{status["label"]}</span>{detail_html}'
        )
    meta_html = (
        f'<div class="yelp-card__meta">{" • ".join(meta_parts)}</div>'
        if meta_parts
        else ""
    )

    tags_html = (
        '<div class="yelp-card__tags">'
        + "".join(f'<span class="yelp-tag">{_escape(c)}</span>' for c in categories)
        + "</div>"
    ) if categories else ""

    return (
        '<div class="yelp-card">'
        f'<div class="yelp-card__name">{name}</div>'
        f'<div class="yelp-card__rating-line">{render_stars(stars)}{score_html}{reviews_html}</div>'
        f'{meta_html}'
        f'{tags_html}'
        '</div>'
    )
