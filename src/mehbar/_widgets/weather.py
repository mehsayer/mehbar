import logging
import time
import urllib.error

import anyio

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.tools import describe_exception, get_cache_home
from mehbar.widget import WidgetBase

from ._weather import (
    PROVIDERS,
    Condition,
    Units,
    Weather,
    WeatherCache,
    compass,
    geocode,
)


class WidgetWeather(WidgetBase):
    """Shows the current weather at `location`, a place name, or at
    `latitude` and `longitude`, from the `provider`. Ramp entries are for
    conditions: clear, partly cloudy, cloudy, fog, drizzle, rain, snow,
    thunderstorm; eight more entries, if given, are for the same conditions
    at night. The widget is hidden until the first forecast arrives.

    Requests are made at most every `FETCH_INTERVAL` seconds, also across
    restarts of the bar: shorter intervals show the cached weather."""

    TYPE = "weather"
    POLLING = True
    UNIQUE = False

    # seconds between requests, failed ones included
    FETCH_INTERVAL = 15 * 60

    FMT_FIELDS = [
        "temp",
        "feels_like",
        "humidity",
        "wind_speed",
        "wind_direction",
        "condition",
        "location",
        "temp_unit",
        "speed_unit",
        "ramp",
    ]

    DEFAULT_TOOLTIP = (
        "{location}: {condition}, {temp}{temp_unit}\n"
        "Feels like {feels_like}{temp_unit}, humidity {humidity}%\n"
        "Wind {wind_speed} {speed_unit} {wind_direction}"
    )

    # network errors, unexpected replies
    FETCH_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError)

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        provider = self.cfg.get("provider", "open-meteo")

        if (provider_cls := PROVIDERS.get(provider)) is None:
            providers = ", ".join(repr(p) for p in PROVIDERS)
            raise BarConfigError(
                f"unknown provider '{provider}', not one of: {providers}"
            )

        api_key = self.cfg.get("api_key")

        if provider_cls.NEEDS_KEY and (not isinstance(api_key, str) or not api_key):
            raise BarConfigError(f"provider '{provider}' needs an 'api_key'")

        units = self.cfg.get("units", "metric")

        if units not in ("metric", "imperial"):
            raise BarConfigError("'units' must be 'metric' or 'imperial'")

        self.units = Units(units == "imperial")
        self.provider = provider_cls(self.units, api_key)

        location = self.cfg.get("location")
        lat, lon = self.cfg.get("latitude"), self.cfg.get("longitude")

        if location is not None:
            if not isinstance(location, str) or not location:
                raise BarConfigError("'location' must be a place name")
            self._coords: tuple[float, float] | None = None
            self._location = location
        elif isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            self._coords = (lat, lon)
            self._location = f"{lat:.2f}, {lon:.2f}"
        else:
            raise BarConfigError(
                "'location', or 'latitude' and 'longitude', must be set"
            )

        if unknown := set(self.label_fields()) - set(self.FMT_FIELDS):
            raise BarConfigError(f"unknown label fields: {', '.join(sorted(unknown))}")

        self.cache = WeatherCache(get_cache_home(), provider, location, lat, lon, units)

        # wall clock time, which counts while the system is suspended
        self._fetched_at = 0.0
        self._weather: Weather | None = None
        self._failing = False
        self.set_available_i(False)

    def ramp_index(self, ramp_level: int) -> int | None:
        if ramp_level < 0 or not self.ramp:
            return None

        # night conditions follow day ones
        if ramp_level >= len(Condition) and len(self.ramp) <= len(Condition):
            ramp_level -= len(Condition)

        return min(ramp_level, len(self.ramp) - 1)

    def _fetch(self) -> Weather:
        """Blocks, call it in a worker thread."""
        if self._coords is None:
            # LookupError if not found, which disables the widget
            lat, lon, self._location = geocode(self._location)
            self._coords = (lat, lon)

        try:
            return self.provider.fetch(*self._coords)
        except urllib.error.HTTPError as ex:
            if ex.code in (401, 403):
                raise BarConfigError(f"the provider rejected the request: {ex}") from ex
            raise

    def _show(self, weather: Weather):
        def _round(value: float | None) -> int | None:
            return None if value is None else round(value)

        ramp_level = int(weather.condition)
        if not weather.is_day:
            ramp_level += len(Condition)

        self.set_new_content_i(
            ramp_level,
            temp=_round(weather.temp),
            feels_like=_round(weather.feels_like),
            humidity=_round(weather.humidity),
            wind_speed=_round(weather.wind_speed),
            wind_direction=compass(weather.wind_direction),
            condition=weather.condition.text,
            location=self._location,
            temp_unit=self.units.temp,
            speed_unit=self.units.speed,
        )

        self.set_available_i(True)

    async def _update(self):
        """Requests the weather, unless the latest request is recent."""
        now = time.time()

        if now - self._fetched_at < self.FETCH_INTERVAL:
            return

        self._fetched_at = now

        try:
            weather = await anyio.to_thread.run_sync(self._fetch)
        except self.FETCH_ERRORS as ex:
            # e.g. offline: the last weather stays until the next request
            if not self._failing:
                self._failing = True
                logging.warning(
                    "widget '%s': cannot get the weather: %s",
                    self.widget_name,
                    describe_exception(ex),
                )
            return

        self._failing = False
        self._weather = weather
        await anyio.to_thread.run_sync(
            self.cache.save, now, weather, self._location, self._coords
        )
        self._show(weather)

    async def run(self):
        if (cached := await anyio.to_thread.run_sync(self.cache.load)) is not None:
            self._fetched_at, self._weather, self._location, self._coords = cached
            self._show(self._weather)

        while await self.sleep_interval():
            await self._update()
