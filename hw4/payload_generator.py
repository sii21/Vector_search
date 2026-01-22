"""Generate random payload fields for testing different filter types"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta
from typing import Any


def generate_random_string(length: int = 10) -> str:
    """Generate random string"""
    return "".join(random.choices(string.ascii_lowercase, k=length))


def generate_keyword_field(choices: list[str] | None = None) -> str:
    """Generate random keyword from predefined list"""
    if choices is None:
        choices = ["red", "green", "blue", "yellow", "black", "white"]
    return random.choice(choices)


def generate_integer_field(min_val: int = 0, max_val: int = 100) -> int:
    """Generate random integer in range"""
    return random.randint(min_val, max_val)


def generate_float_field(min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Generate random float in range"""
    return random.uniform(min_val, max_val)


def generate_bool_field() -> bool:
    """Generate random boolean"""
    return random.choice([True, False])


def generate_geo_point() -> dict[str, float]:
    """Generate random geo point (lat, lon)"""
    return {
        "lat": random.uniform(-90.0, 90.0),
        "lon": random.uniform(-180.0, 180.0),
    }


def generate_datetime_field(
    start_date: datetime | None = None, days_range: int = 365
) -> str:
    """Generate random datetime as ISO string"""
    if start_date is None:
        start_date = datetime(2020, 1, 1)
    random_days = random.randint(0, days_range)
    random_date = start_date + timedelta(days=random_days)
    return random_date.isoformat()


def generate_array_field(
    element_generator: callable, min_length: int = 1, max_length: int = 5
) -> list[Any]:
    """Generate array of random elements"""
    length = random.randint(min_length, max_length)
    return [element_generator() for _ in range(length)]


def generate_nullable_field(generator: callable, null_probability: float = 0.1) -> Any:
    """Generate field that can be null with some probability"""
    if random.random() < null_probability:
        return None
    return generator()


def generate_text_field(word_count: int = 10) -> str:
    """Generate random text for full-text search"""
    words = [
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "cat",
        "bird",
        "fish",
        "tree",
        "house",
        "car",
        "book",
        "computer",
        "phone",
        "table",
        "chair",
    ]
    return " ".join(random.choices(words, k=word_count))


def generate_full_payload() -> dict:
    """Generate complete payload with various field types for testing filters"""
    return {
        "color": generate_keyword_field(["red", "green", "blue", "yellow", "black"]),
        "category": generate_keyword_field(
            ["electronics", "clothing", "food", "books", "toys"]
        ),
        "price": generate_float_field(10.0, 1000.0),
        "quantity": generate_integer_field(0, 100),
        "rating": generate_float_field(1.0, 5.0),
        "in_stock": generate_bool_field(),
        "location": generate_geo_point(),
        "created_at": generate_datetime_field(),
        "tags": generate_array_field(
            lambda: generate_keyword_field(["sale", "new", "popular", "limited"]),
            min_length=0,
            max_length=3,
        ),
        "description": generate_text_field(15),
        "discount": generate_nullable_field(
            lambda: generate_float_field(0.0, 50.0), null_probability=0.3
        ),
        "brand": generate_nullable_field(
            lambda: generate_keyword_field(["BrandA", "BrandB", "BrandC", "BrandD"]),
            null_probability=0.2,
        ),
    }
