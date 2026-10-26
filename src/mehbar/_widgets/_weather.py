"""Weather providers. They are synchronous, call them in a worker thread."""

import enum
import hashlib
import json
import logging
import math
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mehbar import __version__

# MET Norway requires an identifying user agent
USER_AGENT = f"mehbar/{__version__} (+https://github.com/mehsayer/mehbar)"
TIMEOUT = 10

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


class Condition(enum.IntEnum):
    CLEAR = 0
    PARTLY_CLOUDY = 1
    CLOUDY = 2
    FOG = 3
    DRIZZLE = 4
    RAIN = 5
    SNOW = 6
    THUNDERSTORM = 7

    @property
    def text(self) -> str:
        return self.name.replace("_", " ").capitalize()


@dataclass(frozen=True)
class Units:
    imperial: bool

    @property
    def temp(self) -> str:
        return "°F" if self.imperial else "°C"

    @property
    def speed(self) -> str:
        return "mph" if self.imperial else "km/h"

    def temp_from_celsius(self, celsius: float) -> float:
        return celsius * 9 / 5 + 32 if self.imperial else celsius

    def speed_from_mps(self, mps: float) -> float:
        return mps * 2.23694 if self.imperial else mps * 3.6


@dataclass(frozen=True)
class Weather:
    condition: Condition
    is_day: bool
    temp: float
    feels_like: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    # degrees, where the wind comes from
    wind_direction: float | None = None


def apparent_temperature(celsius: float, humidity: float, wind_mps: float) -> float:
    """Steadman's apparent temperature, which Open-Meteo reports as well."""
    vapour = humidity / 100 * 6.105 * math.exp(17.27 * celsius / (237.7 + celsius))
    return celsius + 0.33 * vapour - 0.70 * wind_mps - 4.00


def compass(degrees: float | None) -> str | None:
    if degrees is None:
        return None
    points = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return points[round(degrees / 45) % 8]


def get_json(url: str, params: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


def geocode(name: str) -> tuple[float, float, str]:
    """Returns the latitude, longitude and name of the place."""
    results = get_json(GEOCODING_URL, {"name": name, "count": 1}).get("results")

    if not results:
        raise LookupError(f"location '{name}' not found")

    place = results[0]
    label = ", ".join(p for p in (place.get("name"), place.get("country_code")) if p)

    return place["latitude"], place["longitude"], label


class Provider:
    """`fetch()` raises `OSError` (including network errors), `ValueError`
    (including invalid JSON) or `KeyError` for unexpected replies."""

    NEEDS_KEY = False

    def __init__(self, units: Units, api_key: str | None = None):
        self.units = units
        self.api_key = api_key

    def fetch(self, lat: float, lon: float) -> Weather:
        return self.parse(self.request(lat, lon))

    def request(self, lat: float, lon: float) -> Any:
        raise NotImplementedError()

    def parse(self, data: Any) -> Weather:
        raise NotImplementedError()


class OpenMeteo(Provider):
    """https://open-meteo.com, free for non-commercial use, no key."""

    URL = "https://api.open-meteo.com/v1/forecast"

    # WMO weather interpretation codes
    CONDITIONS = {
        0: Condition.CLEAR,
        1: Condition.PARTLY_CLOUDY,
        2: Condition.PARTLY_CLOUDY,
        3: Condition.CLOUDY,
        45: Condition.FOG,
        48: Condition.FOG,
        **dict.fromkeys((51, 53, 55, 56, 57), Condition.DRIZZLE),
        **dict.fromkeys((61, 63, 65, 66, 67, 80, 81, 82), Condition.RAIN),
        **dict.fromkeys((71, 73, 75, 77, 85, 86), Condition.SNOW),
        **dict.fromkeys((95, 96, 99), Condition.THUNDERSTORM),
    }

    def request(self, lat: float, lon: float) -> Any:
        return get_json(
            self.URL,
            {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "is_day,weather_code,wind_speed_10m,wind_direction_10m",
                "temperature_unit": "fahrenheit" if self.units.imperial else "celsius",
                "wind_speed_unit": "mph" if self.units.imperial else "kmh",
            },
        )

    def parse(self, data: Any) -> Weather:
        current = data["current"]
        return Weather(
            self.CONDITIONS.get(current["weather_code"], Condition.CLOUDY),
            bool(current.get("is_day", 1)),
            current["temperature_2m"],
            current.get("apparent_temperature"),
            current.get("relative_humidity_2m"),
            current.get("wind_speed_10m"),
            current.get("wind_direction_10m"),
        )


class MetNorway(Provider):
    """https://api.met.no, the Norwegian Meteorological Institute, free, no
    key; it asks for at most 4 decimals in coordinates."""

    URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

    def request(self, lat: float, lon: float) -> Any:
        return get_json(self.URL, {"lat": round(lat, 4), "lon": round(lon, 4)})

    @staticmethod
    def condition(symbol: str) -> Condition:
        # e.g. 'partlycloudy_day', 'lightrainshowers_night', 'heavysnow'
        symbol = symbol.split("_")[0]

        if "thunder" in symbol:
            return Condition.THUNDERSTORM
        if "snow" in symbol or "sleet" in symbol:
            return Condition.SNOW
        if symbol.startswith("light") and "rain" in symbol:
            return Condition.DRIZZLE
        if "rain" in symbol:
            return Condition.RAIN
        if symbol == "fog":
            return Condition.FOG
        if symbol == "cloudy":
            return Condition.CLOUDY
        if symbol in ("fair", "partlycloudy"):
            return Condition.PARTLY_CLOUDY
        return Condition.CLEAR

    def parse(self, data: Any) -> Weather:
        series = data["properties"]["timeseries"][0]["data"]
        details = series["instant"]["details"]
        symbol = (
            (series.get("next_1_hours") or series.get("next_6_hours") or {})
            .get("summary", {})
            .get("symbol_code", "cloudy")
        )

        celsius = details["air_temperature"]
        humidity = details.get("relative_humidity")
        wind_mps = details.get("wind_speed")

        feels_like = None
        if humidity is not None and wind_mps is not None:
            feels_like = self.units.temp_from_celsius(
                apparent_temperature(celsius, humidity, wind_mps)
            )

        return Weather(
            self.condition(symbol),
            not symbol.endswith("_night"),
            self.units.temp_from_celsius(celsius),
            feels_like,
            humidity,
            None if wind_mps is None else self.units.speed_from_mps(wind_mps),
            details.get("wind_from_direction"),
        )


class OpenWeatherMap(Provider):
    """https://openweathermap.org, needs an API key, which is free for up to
    1000 calls a day."""

    URL = "https://api.openweathermap.org/data/2.5/weather"
    NEEDS_KEY = True

    @staticmethod
    def condition(code: int) -> Condition:
        match code // 100:
            case 2:
                return Condition.THUNDERSTORM
            case 3:
                return Condition.DRIZZLE
            case 5:
                return Condition.RAIN
            case 6:
                return Condition.SNOW
            case 7:
                return Condition.FOG
        if code == 800:
            return Condition.CLEAR
        if code in (801, 802):
            return Condition.PARTLY_CLOUDY
        return Condition.CLOUDY

    def request(self, lat: float, lon: float) -> Any:
        return get_json(
            self.URL,
            {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "imperial" if self.units.imperial else "metric",
            },
        )

    def parse(self, data: Any) -> Weather:
        main = data["main"]
        wind = data.get("wind", {})
        weather = data["weather"][0]
        speed = wind.get("speed")

        # m/s in metric units, mph in imperial ones
        if speed is not None and not self.units.imperial:
            speed = self.units.speed_from_mps(speed)

        return Weather(
            self.condition(weather["id"]),
            weather.get("icon", "d").endswith("d"),
            main["temp"],
            main.get("feels_like"),
            main.get("humidity"),
            speed,
            wind.get("deg"),
        )


class WeatherCache:
    """The latest weather of a location, saved to a file, so that restarts
    of the bar do not repeat requests."""

    def __init__(self, directory: Path, *key: Any):
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        self.path = directory / f"weather-{digest}.json"

    def load(self) -> tuple[float, Weather, str, tuple[float, float]] | None:
        """Returns the time of the request, the weather, the location name
        and coordinates."""
        try:
            data = json.loads(self.path.read_text())
            lat, lon = data["coords"]
            weather = Weather(
                **(
                    data["weather"]
                    | {"condition": Condition(data["weather"]["condition"])}
                )
            )
            return data["time"], weather, data["location"], (lat, lon)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(
        self,
        time: float,
        weather: Weather,
        location: str,
        coords: tuple[float, float] | None,
    ):
        data = {
            "time": time,
            "weather": asdict(weather),
            "location": location,
            "coords": coords,
        }
        tmp_path = self.path.with_suffix(".tmp")

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(data))
            os.replace(tmp_path, self.path)
        except OSError as ex:
            logging.debug("cannot save the weather to '%s': %s", self.path, ex)


PROVIDERS: dict[str, type[Provider]] = {
    "open-meteo": OpenMeteo,
    "met.no": MetNorway,
    "openweathermap": OpenWeatherMap,
}
