from superlinked import framework as sl
from api.agents.superlinked_app.index import (
    business_index, 
    review_count_space, 
    review_rating_space, 
    categories_space,
    business
)

from api.core.config import config
from collections import namedtuple
from api.agents.superlinked_app.utils.utils import *


openai_config = sl.OpenAIClientConfig(
    api_key=config.openai_api_key,
    model="gpt-4o-mini",
)


query = (
    sl.Query(
        business_index,
        weights={
            review_count_space: sl.Param(
                "review_count_weight"
            ),
            review_rating_space: sl.Param(
                "rreview_rating_weight"
            ),
            categories_space: sl.Param("categories_weight"),
        },
    )
    .find(business)
    .similar(
        categories_space.text,
        sl.Param(
            "description",
            description=(
                "Semantic text for the business category field: city, neighborhood, vibe, atmosphere, "
                "dietary or service hints. Omit repeating the venue type; use required_category_tags for that."
            ),
        ),
    )
)

# We can specify number of retreved results like this:
query = query.limit(sl.Param("limit", default=4))

# We want all fields to be returned
query = query.select_all()

# .. and all the metadata including knn_params and partial_scores
query = query.include_metadata()

# Now let's add hard-filtering
# for city:
query = query.filter(
    business.city.in_(sl.Param("city", description="used to filter by city"))
)

# ... for numerical attributes:
query = (
    query
    .filter(business.stars >= sl.Param("min_rating"))
    .filter(business.stars <= sl.Param("max_rating"))
)

# ... and for all categorical attributes:
CategoryFilter = namedtuple(
    "CategoryFilter", ["operator", "param_name", "category_name", "description"]
)

filters = [
    CategoryFilter(
        operator=lambda p: business.category_tags.contains_all(p),
        param_name="required_category_tags",
        category_name="required_category_tags",
        description=(
            "List of Yelp category strings the business must have (e.g. ['Cocktail Bars'] or "
            "['Restaurants', 'Italian']). Match Yelp spelling/casing. Use null if the user does not name a venue type."
        ),
    ),
    CategoryFilter(
        operator=lambda p: business.is_open_i == p,
        param_name="is_open_i",
        category_name="is_open_i",
        description="Open status: 1 = open, 0 = closed. Use null if not specified.",
    ),
    CategoryFilter(
        operator=lambda p: business.bike_parking_i == p,
        param_name="bike_parking_i",
        category_name="bike_parking_i",
        description="1 = must have bike parking, 0 = must not, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.accepts_credit_cards_i == p,
        param_name="accepts_credit_cards_i",
        category_name="accepts_credit_cards_i",
        description="1 = accepts cards, 0 = does not, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.price_range == p,
        param_name="price_range",
        category_name="price_range",
        description="Yelp price level 1–4; use null if not specified.",
    ),
    CategoryFilter(
        operator=lambda p: business.wifi == p,
        param_name="wifi",
        category_name="wifi",
        description="WiFi: no, free, paid, or unknown; null if not specified.",
    ),
    CategoryFilter(
        operator=lambda p: business.parking_garage_i == p,
        param_name="parking_garage_i",
        category_name="parking_garage_i",
        description="1 = garage parking, 0 = no garage, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.parking_lot_i == p,
        param_name="parking_lot_i",
        category_name="parking_lot_i",
        description="1 = lot parking, 0 = no lot, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.parking_street_i == p,
        param_name="parking_street_i",
        category_name="parking_street_i",
        description="1 = street parking, 0 = no street, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.parking_valet_i == p,
        param_name="parking_valet_i",
        category_name="parking_valet_i",
        description="1 = valet, 0 = no valet, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.parking_validated_i == p,
        param_name="parking_validated_i",
        category_name="parking_validated_i",
        description="1 = validated parking, 0 = no, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.by_appointment_only_i == p,
        param_name="by_appointment_only_i",
        category_name="by_appointment_only_i",
        description="1 = by appointment only, 0 = walk-in ok, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.outdoor_seating_i == p,
        param_name="outdoor_seating_i",
        category_name="outdoor_seating_i",
        description="1 = outdoor seating, 0 = no, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.wheelchair_accessible_i == p,
        param_name="wheelchair_accessible_i",
        category_name="wheelchair_accessible_i",
        description="1 = wheelchair accessible, 0 = not, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.happy_hour_i == p,
        param_name="happy_hour_i",
        category_name="happy_hour_i",
        description="1 = has happy hour, 0 = no, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.takes_out_i == p,
        param_name="takes_out_i",
        category_name="takes_out_i",
        description="1 = takeout, 0 = no takeout, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.reservations_i == p,
        param_name="reservations_i",
        category_name="reservations_i",
        description="1 = takes reservations, 0 = no, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.delivery_i == p,
        param_name="delivery_i",
        category_name="delivery_i",
        description="1 = delivery, 0 = no delivery, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.has_tv_i == p,
        param_name="has_tv_i",
        category_name="has_tv_i",
        description="1 = has TV, 0 = no, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda p: business.dogs_allowed_i == p,
        param_name="dogs_allowed_i",
        category_name="dogs_allowed_i",
        description="1 = dogs allowed, 0 = not, null = ignore.",
    ),
    CategoryFilter(
        operator=lambda t: [
            business.mon_open <= t,
            business.mon_close >= t,
        ],
        param_name="mon_time",
        category_name="mon_time",
        description="Minutes since midnight (0–1439) on Monday; business must be open at that time. Null = ignore.",
    ),
    CategoryFilter(
        operator=lambda t: [
            business.tue_open <= t,
            business.tue_close >= t,
        ],
        param_name="tue_time",
        category_name="tue_time",
        description="Minutes since midnight on Tuesday; null = ignore.",
    ),
    CategoryFilter(
        operator=lambda t: [
            business.wed_open <= t,
            business.wed_close >= t,
        ],
        param_name="wed_time",
        category_name="wed_time",
        description="Minutes since midnight on Wednesday; null = ignore.",
    ),
    CategoryFilter(
        operator=lambda t: [
            business.thu_open <= t,
            business.thu_close >= t,
        ],
        param_name="thu_time",
        category_name="thu_time",
        description="Minutes since midnight on Thursday; null = ignore.",
    ),
    CategoryFilter(
        operator=lambda t: [
            business.fri_open <= t,
            business.fri_close >= t,
        ],
        param_name="fri_time",
        category_name="fri_time",
        description="Minutes since midnight on Friday; null = ignore.",
    ),
    CategoryFilter(
        operator=lambda t: [
            business.sat_open <= t,
            business.sat_close >= t,
        ],
        param_name="sat_time",
        category_name="sat_time",
        description="Minutes since midnight on Saturday; null = ignore.",
    ),
    CategoryFilter(
        operator=lambda t: [
            business.sun_open <= t,
            business.sun_close >= t,
        ],
        param_name="sun_time",
        category_name="sun_time",
        description="Minutes since midnight on Sunday; null = ignore.",
    ),
]

for filter_item in filters:
    param = sl.Param(
        filter_item.param_name,
        description=filter_item.description,
        # options= cat_options[filter_item.category_name],
    )

    ops = filter_item.operator(param)
    if isinstance(ops, list):
        for op in ops:
            query = query.filter(op)
    else:
        query = query.filter(ops)


system_prompt = (
    "Extract the search parameters from the user query.\n"
    "Advices:\n"
    "**required_category_tags** — When the user names a venue or business type, set this to a list of Yelp-style "
    "category labels that capture that type (e.g. ['Cocktail Bars'], ['Mexican'], ['Restaurants', 'Italian']). "
    "Every listed tag must appear on the business. Use common Yelp names and Title Case. Use null if they do not "
    "name a specific type.\n"
    "**description** — Free text for semantic match on stored category text: location, neighborhood, vibe, "
    "atmosphere, dietary notes. Do not repeat the venue-type wording here; that belongs in required_category_tags.\n"
    "**is_open_i** — 1 = open only, 0 = closed only, null = ignore.\n"
    "Params ending in _i are 0/1 amenity flags: 1 = must have, 0 = must not, null = ignore.\n"
    "**wifi** — one of: no, free, paid, unknown; null if not specified.\n"
    "**price_range** — Yelp dollar level 1–4; null if not specified.\n"
    "**mon_time** … **sun_time** — minutes since midnight (e.g. 720 = noon) for that weekday; null unless the user asks about being open at a specific time that day.\n"
    # "**'include' and 'exclude' attributes**\n"
    # "Use relevant amenities, for example, include 'Cot' when user mentions 'baby',"
    # "and exclude it when user mentions 'no children'.\n"
    # "If no amenities are mentioned, use None for 'include' and 'exclude'.\n"
    # "**'accomodation_type'**\n"
    # "If users searches for some restaurants, include 'Restaurants' in categories types, "
    # "same for other accomodation types.\n"
)

# And finally, let's add natural language interface on top
# that will call LLM to parse user natural query
# into structured superlinked query, i.e. suggest parameters values.
query = query.with_natural_query(
    natural_query=sl.Param("natural_query"),
    client_config=openai_config,
    system_prompt=system_prompt,
)
