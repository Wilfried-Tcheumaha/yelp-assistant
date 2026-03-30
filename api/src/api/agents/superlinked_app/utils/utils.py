from superlinked import framework as sl
from api.core.config import config
import pandas as pd
import json
import ast
import re
from typing import Any, Optional


# ----------------------------
# Helpers (normalize Yelp fields)
# ----------------------------

_TRUE_SET = {"True","true", "1", "yes", "y", "t"}
_FALSE_SET = {"False","false", "0", "no", "n", "f"}

def parse_bool(v: Any) -> Optional[bool]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in _TRUE_SET:
        return True
    if s in _FALSE_SET:
        return False
    return None


def bool_for_index(v: Any) -> bool:
    """Yelp omits many attributes; Index fields must be non-null on ingest."""
    b = parse_bool(v)
    return False if b is None else b


def parse_int(v: Any) -> Optional[int]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(str(v).strip())
    except Exception:
        return None

_wifi_pat = re.compile(r"^u'(.+)'$")  # converts "u'no'" -> "no"
def normalize_wifi(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    m = _wifi_pat.match(s)
    if m:
        s = m.group(1)
    s = s.strip().strip('"').strip("'").lower()
    if s in {"no", "free", "paid"}:
        return s
    return "unknown"

def parse_categories_list(v: Any) -> list[str]:
    # Yelp "categories" commonly comes as "A, B, C"
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]

def categories_text_from_list(cats: list[str]) -> str:
    # a single string for TextSimilaritySpace input
    return ", ".join(cats)

def parse_business_parking(v: Any) -> dict:
    # Yelp often stores this as a stringified python dict
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return {}
    if isinstance(v, dict):
        return v
    s = str(v).strip()
    if not s:
        return {}
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}

def parse_time_to_minutes(hhmm: str) -> Optional[int]:
    # handles "8:0" or "08:00"
    if hhmm is None:
        return None
    s = str(hhmm).strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) != 2:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        if 0 <= h <= 24 and 0 <= m < 60:
            return h * 60 + m
    except Exception:
        return None
    return None

def parse_hours_range(v: Any) -> tuple[Optional[int], Optional[int], Optional[bool]]:
    # e.g. "8:0-22:0" -> (480, 1320, overnight=False)
    if v is None:
        return (None, None, None)
    s = str(v).strip()
    if not s or s == "0:0-0:0":
        return (None, None, None)
    if "-" not in s:
        return (None, None, None)
    start_s, end_s = [p.strip() for p in s.split("-", 1)]
    start = parse_time_to_minutes(start_s)
    end = parse_time_to_minutes(end_s)
    if start is None or end is None:
        return (None, None, None)
    overnight = end < start
    if overnight:
        end = end + 1440  # represent "next day" close
    return (start, end, overnight)


def minutes_to_hhmm(m: Any, *, zero_display: str = "—") -> str:
    """Format minutes since midnight for display (allows hours > 23 after overnight shift)."""
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return zero_display
    try:
        mi = int(m)
    except (TypeError, ValueError):
        return zero_display
    if mi == 0:
        return zero_display
    h, mm = divmod(mi, 60)
    return f"{h:d}:{mm:02d}"


_MINUTE_TIME_COLUMNS = tuple(
    f"{day}_{kind}" for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun") for kind in ("open", "close")
)


def format_minute_columns_to_hhmm(df: pd.DataFrame) -> pd.DataFrame:
    """Convert indexed weekday open/close minute fields to hh:mm strings for display."""
    out = df.copy()
    for col in _MINUTE_TIME_COLUMNS:
        if col in out.columns:
            out[col] = out[col].apply(minutes_to_hhmm)
    return out



# ----------------------------
# DataFrame -> schema mapping
# ----------------------------

def get_attr(row: pd.Series, key: str) -> Any:
    attrs = row.get("attributes")
    if isinstance(attrs, dict):
        return attrs.get(key)
    return None


def get_hours(row: pd.Series, day: str) -> Any:
    hours = row.get("hours")
    if isinstance(hours, dict):
        return hours.get(day)
    return None


def int01(v: Any) -> int:
    return 1 if bool_for_index(v) else 0


# DataFrameParser mapping values must be column names (str). Callable values are not
# row-wise transforms: pandas treats a callable indexer as df.apply(lambda), so the
# callable receives the whole DataFrame. Build a flat frame whose columns match schema
# field names, then use the default parser (no mapping).


def business_row_for_index(row: pd.Series) -> pd.Series:
    def _str_cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v).strip()

    def _float_cell(v: Any, default: float = 0.0) -> float:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)

    attrs_raw = row.get("attributes")
    attributes_s = (
        json.dumps(attrs_raw, sort_keys=True, default=str) if isinstance(attrs_raw, dict) else "{}"
    )
    park = parse_business_parking(get_attr(row, "BusinessParking"))
    hours_raw = row.get("hours")
    hours_s = json.dumps(hours_raw, sort_keys=True, default=str) if isinstance(hours_raw, dict) else "{}"
    is_open_b = bool(int(row["is_open"])) if pd.notna(row.get("is_open")) else False

    def oc(day: str) -> tuple[int, int]:
        o, c, _ = parse_hours_range(get_hours(row, day))
        return (o or 0, c or 0)

    mo_o, mo_c = oc("Monday")
    tu_o, tu_c = oc("Tuesday")
    we_o, we_c = oc("Wednesday")
    th_o, th_c = oc("Thursday")
    fr_o, fr_c = oc("Friday")
    sa_o, sa_c = oc("Saturday")
    su_o, su_c = oc("Sunday")
    cats = parse_categories_list(row.get("categories"))

    return pd.Series(
        {
            "business_id": row["business_id"],
            "name": _str_cell(row.get("name")),
            "address": _str_cell(row.get("address")),
            "city": _str_cell(row.get("city")),
            "state": _str_cell(row.get("state")),
            "postal_code": _str_cell(row.get("postal_code")),
            "latitude": _float_cell(row.get("latitude")),
            "longitude": _float_cell(row.get("longitude")),
            "stars": _float_cell(row.get("stars")),
            "review_count": int(row["review_count"]) if pd.notna(row.get("review_count")) else 0,
            "is_open": is_open_b,
            "is_open_i": 1 if is_open_b else 0,
            "attributes": attributes_s,
            "bike_parking_i": int01(get_attr(row, "BikeParking")),
            "accepts_credit_cards_i": int01(get_attr(row, "BusinessAcceptsCreditCards")),
            "price_range": parse_int(get_attr(row, "RestaurantsPriceRange2")) or 0,
            "wifi": normalize_wifi(get_attr(row, "WiFi")) or "unknown",
            "parking_garage_i": int01(park.get("garage")),
            "parking_street_i": int01(park.get("street")),
            "parking_validated_i": int01(park.get("validated")),
            "parking_lot_i": int01(park.get("lot")),
            "parking_valet_i": int01(park.get("valet")),
            "wheelchair_accessible_i": int01(get_attr(row, "WheelchairAccessible")),
            "happy_hour_i": int01(get_attr(row, "HappyHour")),
            "outdoor_seating_i": int01(get_attr(row, "OutdoorSeating")),
            "has_tv_i": int01(get_attr(row, "HasTV")),
            "takes_out_i": int01(get_attr(row, "RestaurantsTakeOut")),
            "delivery_i": int01(get_attr(row, "RestaurantsDelivery")),
            "reservations_i": int01(get_attr(row, "RestaurantsReservations")),
            "dogs_allowed_i": int01(get_attr(row, "DogsAllowed")),
            "by_appointment_only_i": int01(get_attr(row, "ByAppointmentOnly")),
            "category_tags": cats,
            "categories_text": categories_text_from_list(cats),
            "hours": hours_s,
            "mon_open": mo_o,
            "mon_close": mo_c,
            "tue_open": tu_o,
            "tue_close": tu_c,
            "wed_open": we_o,
            "wed_close": we_c,
            "thu_open": th_o,
            "thu_close": th_c,
            "fri_open": fr_o,
            "fri_close": fr_c,
            "sat_open": sa_o,
            "sat_close": sa_c,
            "sun_open": su_o,
            "sun_close": su_c,
        }
    )

