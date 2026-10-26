import pytest

from mehbar._widgets._weather import (
    Condition,
    MetNorway,
    OpenMeteo,
    OpenWeatherMap,
    Units,
    apparent_temperature,
    compass,
)

METRIC, IMPERIAL = Units(False), Units(True)


def test_open_meteo():
    data = {
        "current": {
            "temperature_2m": 12.5,
            "apparent_temperature": 9.4,
            "relative_humidity_2m": 62,
            "is_day": 0,
            "weather_code": 61,
            "wind_speed_10m": 13.2,
            "wind_direction_10m": 244,
        }
    }
    weather = OpenMeteo(METRIC).parse(data)

    assert weather.condition == Condition.RAIN
    assert not weather.is_day
    assert (weather.temp, weather.feels_like, weather.humidity) == (12.5, 9.4, 62)
    assert (weather.wind_speed, weather.wind_direction) == (13.2, 244)


def test_met_norway_converts_units():
    data = {
        "properties": {
            "timeseries": [
                {
                    "data": {
                        "instant": {
                            "details": {
                                "air_temperature": 10.0,
                                "relative_humidity": 50.0,
                                "wind_speed": 10.0,
                                "wind_from_direction": 180.0,
                            }
                        },
                        "next_1_hours": {
                            "summary": {"symbol_code": "partlycloudy_night"}
                        },
                    }
                }
            ]
        }
    }

    metric = MetNorway(METRIC).parse(data)
    imperial = MetNorway(IMPERIAL).parse(data)

    assert metric.condition == Condition.PARTLY_CLOUDY
    assert not metric.is_day
    assert metric.wind_speed == pytest.approx(36)
    assert imperial.temp == pytest.approx(50)
    assert imperial.wind_speed == pytest.approx(22.37, abs=0.01)
    # computed, MET Norway does not report it
    assert metric.feels_like == pytest.approx(apparent_temperature(10, 50, 10))


@pytest.mark.parametrize(
    ("symbol", "condition"),
    [
        ("clearsky_day", Condition.CLEAR),
        ("fair_night", Condition.PARTLY_CLOUDY),
        ("cloudy", Condition.CLOUDY),
        ("fog", Condition.FOG),
        ("lightrainshowers_day", Condition.DRIZZLE),
        ("heavyrain", Condition.RAIN),
        ("sleetshowers_day", Condition.SNOW),
        ("rainandthunder", Condition.THUNDERSTORM),
    ],
)
def test_met_norway_symbols(symbol, condition):
    assert MetNorway.condition(symbol) == condition


def test_openweathermap_converts_metric_wind():
    data = {
        "main": {"temp": 20.0, "feels_like": 19.0, "humidity": 40},
        "wind": {"speed": 5.0, "deg": 90},
        "weather": [{"id": 802, "icon": "03d"}],
    }

    weather = OpenWeatherMap(METRIC, "key").parse(data)

    assert weather.condition == Condition.PARTLY_CLOUDY
    assert weather.is_day
    assert weather.wind_speed == pytest.approx(18)
    # mph in imperial units already
    assert OpenWeatherMap(IMPERIAL, "key").parse(data).wind_speed == 5.0


@pytest.mark.parametrize(
    ("code", "condition"),
    [
        (211, Condition.THUNDERSTORM),
        (301, Condition.DRIZZLE),
        (502, Condition.RAIN),
        (601, Condition.SNOW),
        (741, Condition.FOG),
        (800, Condition.CLEAR),
        (804, Condition.CLOUDY),
    ],
)
def test_openweathermap_codes(code, condition):
    assert OpenWeatherMap.condition(code) == condition


@pytest.mark.parametrize(
    ("degrees", "point"), [(None, None), (0, "N"), (44, "NE"), (180, "S"), (350, "N")]
)
def test_compass(degrees, point):
    assert compass(degrees) == point


@pytest.mark.display
def test_weather_widget_options(make_res_mgr):
    from mehbar._widgets.weather import WidgetWeather
    from mehbar.exceptions import BarConfigError

    def _make(options):
        res_mgr = make_res_mgr(f'[end.w]\ntype = "weather"\ninterval = 900\n{options}')
        return WidgetWeather("w", res_mgr)

    widget = _make("latitude = 52.5\nlongitude = 13.4\nramp = ['d0', 'd1']")
    # night conditions use the day entries without night ones
    assert widget.ramp_index(len(Condition) + 1) == 1

    with pytest.raises(BarConfigError, match="api_key"):
        _make('location = "Berlin"\nprovider = "openweathermap"')
    with pytest.raises(BarConfigError, match="location"):
        _make("")
    with pytest.raises(BarConfigError, match="unknown label fields"):
        _make('location = "Berlin"\nlabel = "{pressure}"')


@pytest.mark.display
def test_weather_requests_are_cached(make_res_mgr, tmp_path, monkeypatch):
    import anyio

    from mehbar._widgets import weather as weather_module
    from mehbar._widgets._weather import Weather

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    requests = []

    def _fetch(self, lat, lon):
        requests.append((lat, lon))
        return Weather(Condition.CLEAR, True, 20.0)

    monkeypatch.setattr(OpenMeteo, "fetch", _fetch)
    clock = [1000.0]
    monkeypatch.setattr(weather_module.time, "time", lambda: clock[0])

    def _widget():
        res_mgr = make_res_mgr(
            '[end.w]\ntype = "weather"\ninterval = 60\nlatitude = 47.27\nlongitude = 11.39'
        )
        return weather_module.WidgetWeather("w", res_mgr)

    widget = _widget()
    anyio.run(widget._update)
    clock[0] += 60
    anyio.run(widget._update)
    assert len(requests) == 1

    clock[0] += 15 * 60
    anyio.run(widget._update)
    assert len(requests) == 2

    # a restarted bar uses the cached weather
    restarted = _widget()
    cached = restarted.cache.load()
    assert cached is not None and cached[1].temp == 20.0
