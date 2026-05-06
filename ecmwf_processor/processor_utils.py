from datetime import datetime, timedelta
import numpy as np
from ecmwf_processor import constants as c
from ecmwf_processor import processor_constants as cc

from typing import Any
import xarray as xr


def get_grib2_to_final_renaming(
    processor_kwargs: dict[str, Any],
) -> dict[str, str]:
    grib2_to_final_renaming: dict = {
        v[c.formula]: k
        for k, v in processor_kwargs[c.bands_and_indices_specs][
            c.ecmwf_ifs
        ].items()
    }

    return grib2_to_final_renaming


def round_up_to_12h_interval(datetime: datetime) -> datetime:
    if (
        datetime.minute == datetime.second == datetime.microsecond == 0
        and datetime.hour in (0, 12)
    ):
        return datetime.replace(tzinfo=None)
    elif datetime.hour > 12:
        return datetime(
            datetime.year, datetime.month, datetime.day, 0, 0
        ) + timedelta(days=1)
    else:
        return datetime(datetime.year, datetime.month, datetime.day, 12, 0)


def round_down_to_12h_interval(datetime: datetime) -> datetime:
    if (
        datetime.minute == datetime.second == datetime.microsecond == 0
        and datetime.hour in (0, 12)
    ):
        return datetime.replace(tzinfo=None)
    elif datetime.hour > 12:
        return datetime(datetime.year, datetime.month, datetime.day, 12, 0)
    else:
        return datetime(datetime.year, datetime.month, datetime.day, 0, 0)


def generate_steps(step_kwargs: dict[str, int]) -> xr.DataArray:
    min_step = step_kwargs[c.min]
    max_step = step_kwargs[c.max]

    filtered_steps = [
        level for level in cc.ifs_steps360 if min_step <= level <= max_step
    ]

    timedelta_steps = [
        np.timedelta64(h, "h").astype("timedelta64[ns]")
        for h in filtered_steps
    ]

    steps = xr.DataArray(
        data=timedelta_steps,
        dims=[c.step],
        coords={c.step: filtered_steps},
        name=c.step,
    )

    return steps


def generate_levels(level_kwargs: dict[str, int]) -> xr.DataArray:

    min_level = level_kwargs[c.min]
    max_level = level_kwargs[c.max]

    filtered_levels = [
        level
        for level in cc.levels_20240201
        if min_level <= level <= max_level
    ]
    return xr.DataArray(
        data=filtered_levels,
        dims=[c.level],
        coords={c.level: filtered_levels},
        name=c.level,
        attrs={c.long_name: c.level, c.units: c.hectopascal},
    )


def generate_soil_layers(soil_layer_kwargs: dict[str, int]) -> xr.DataArray:
    min_layer = soil_layer_kwargs[c.min]
    max_layer = soil_layer_kwargs[c.max]
    filtered_layers = [
        layer for layer in cc.soil_layers if min_layer <= layer <= max_layer
    ]
    return xr.DataArray(
        data=filtered_layers,
        dims=[c.soil_layer],
        coords={c.soil_layer: filtered_layers},
        name=c.soil_layer,
        attrs={c.long_name: c.soil_layer},
    )
