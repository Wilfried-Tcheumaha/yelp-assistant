from superlinked import framework as sl
from api.core.config import config
from api.agents.superlinked_app.utils.utils import *


# ----------------------------
# Superlinked schema — same fields / order as df_businesses (Yelp business JSON).
# attributes & hours are dicts in pandas; indexed as stable JSON strings for Superlinked.
# `category_tags` = parsed Yelp labels (exact strings for filters); `categories_text` = joined for embeddings only.
# ----------------------------

class Business(sl.Schema):
    business_id: sl.IdField
    name: sl.String
    address: sl.String
    city: sl.String
    state: sl.String
    postal_code: sl.String
    latitude: sl.Float
    longitude: sl.Float
    stars: sl.Float
    review_count: sl.Integer
    is_open: sl.Boolean
    is_open_i: sl.Integer  # 1/0 mirror for queries + NLQ (bool params are not NLQ-supported)
    attributes: sl.String
    # Parsed from Yelp `attributes` dict; use *_i (0/1) so NLQ can fill int params (bool is unsupported).
    bike_parking_i: sl.Integer
    accepts_credit_cards_i: sl.Integer
    price_range: sl.Integer
    wifi: sl.String
    parking_garage_i: sl.Integer
    parking_street_i: sl.Integer
    parking_validated_i: sl.Integer
    parking_lot_i: sl.Integer
    parking_valet_i: sl.Integer
    wheelchair_accessible_i: sl.Integer
    happy_hour_i: sl.Integer
    outdoor_seating_i: sl.Integer
    has_tv_i: sl.Integer
    takes_out_i: sl.Integer
    delivery_i: sl.Integer
    reservations_i: sl.Integer
    dogs_allowed_i: sl.Integer
    by_appointment_only_i: sl.Integer
    category_tags: sl.StringList  # Yelp labels (exact strings) for contains_all filter
    categories_text: sl.String  # comma-joined; TextSimilaritySpace input only
    hours: sl.String
    # Parsed from Yelp `hours` dict (minutes since midnight; close may exceed 1440 if overnight)
    mon_open: sl.Integer
    mon_close: sl.Integer
    tue_open: sl.Integer
    tue_close: sl.Integer
    wed_open: sl.Integer
    wed_close: sl.Integer
    thu_open: sl.Integer
    thu_close: sl.Integer
    fri_open: sl.Integer
    fri_close: sl.Integer
    sat_open: sl.Integer
    sat_close: sl.Integer
    sun_open: sl.Integer
    sun_close: sl.Integer


business = Business()

# categories text similarity space
categories_space = sl.TextSimilaritySpace(
    text=business.categories_text,
    model="sentence-transformers/all-MiniLM-L6-v2",
)


# NumberSpaces encode numeric input in special ways to reflect a relationship
# here we express relationships to price (lower the better), or ratings and review counts (more/higher the better)
review_count_space = sl.NumberSpace(number=business.review_count, mode=sl.Mode.MAXIMUM, scale=sl.LogarithmicScale(), min_value=5, max_value=1359)
review_rating_space = sl.NumberSpace(number=business.stars, mode=sl.Mode.MAXIMUM, min_value=1, max_value=5)
 
#Derive the index
business_index = sl.Index(
    spaces=[
        categories_space,
        review_count_space,
        review_rating_space,
        # Add other spaces as needed if defined, like embedding or geo spaces
    ],
    fields=[
        business.name,
        business.address,
        business.city,
        business.state,
        business.postal_code,
        business.latitude,
        business.longitude,
        business.stars,
        business.review_count,
        business.is_open,
        business.is_open_i,
        business.attributes,
        business.bike_parking_i,
        business.accepts_credit_cards_i,
        business.price_range,
        business.wifi,
        business.parking_garage_i,
        business.parking_street_i,
        business.parking_validated_i,
        business.parking_lot_i,
        business.parking_valet_i,
        business.wheelchair_accessible_i,
        business.happy_hour_i,
        business.outdoor_seating_i,
        business.has_tv_i,
        business.takes_out_i,
        business.delivery_i,
        business.reservations_i,
        business.dogs_allowed_i,
        business.by_appointment_only_i,
        business.category_tags,
        business.categories_text,
        business.hours,
        business.mon_open,
        business.mon_close,
        business.tue_open,
        business.tue_close,
        business.wed_open,
        business.wed_close,
        business.thu_open,
        business.thu_close,
        business.fri_open,
        business.fri_close,
        business.sat_open,
        business.sat_close,
        business.sun_open,
        business.sun_close,
    ],
)

