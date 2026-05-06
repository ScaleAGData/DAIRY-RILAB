from datetime import datetime
from typing import Any

import pandas as pd
import zarr
import xarray as xr

from ecmwf_processor import constants as c
from ecmwf_processor.process_options import PROCESSOR_KWARGS
from ecmwf_processor import geometry, zarr_ops
from ecmwf_processor.config import SETTINGS
from ecmwf_processor import processor_constants as cc
from ecmwf_processor import processor_utils


def register_aoi(
    bbox: list[float],
    prefix: str = "preprocessor_ecmwf_ifs",
) -> tuple[xr.Dataset, zarr.storage.Store]:

    cube, cube_store = zarr_ops.initialize_dst_store(
        coords=generate_coords(
            dim_kwargs=PROCESSOR_KWARGS[c.cube_dimensions],
            bbox=geometry.compute_ecmwf_era5_adjusted_bbox(
                bbox=bbox,
                bbox_px_padding=PROCESSOR_KWARGS.get(c.bbox_px_padding, 0),
            ),
        ),
        dim_order=cc.xcube_dim_order,
        src_var_dims={
            **cc.grib2_var_dims_20240201,
            **cc.grib2_var_dims_latest,
        },
        src_var_attrs={
            **cc.grib2_var_attrs_20240201,
            **cc.grib2_var_attrs_latest,
        },
        main_vars_to_specs=PROCESSOR_KWARGS[c.bands_and_indices_specs][
            c.ecmwf_ifs
        ],
        src_to_final_renaming=processor_utils.get_grib2_to_final_renaming(
            PROCESSOR_KWARGS
        ),
        prefix=f"{prefix}",
        dst_chunks=PROCESSOR_KWARGS[c.dst_chunking],
        xr_open_zarr_kwargs={
            c.consolidated: False,
            c.mask_and_scale: False,
        },
        clear_store_before_run=PROCESSOR_KWARGS.get(
            c.delete_everything_before_run, False
        ),
    )

    return cube, cube_store


def generate_coords(
    dim_kwargs: dict[str, dict[str, Any]],
    bbox: list[float],
) -> dict[str, xr.DataArray]:

    coords: dict[str, xr.DataArray] = {
        c.time: generate_time_coord(dim_kwargs=dim_kwargs),
        c.latitude: generate_latitude_coord(bbox=bbox),
        c.longitude: generate_longitude_coord(bbox=bbox),
        c.step: processor_utils.generate_steps(step_kwargs=dim_kwargs[c.step]),
    }

    if c.level in dim_kwargs:
        coords[c.level] = processor_utils.generate_levels(
            level_kwargs=dim_kwargs[c.level]
        )

    if c.soil_layer in dim_kwargs:
        coords[c.soil_layer] = processor_utils.generate_soil_layers(
            soil_layer_kwargs=dim_kwargs[c.soil_layer]
        )

    return coords


def generate_time_coord(dim_kwargs: dict[str, Any]) -> xr.DataArray:
    start_dt = processor_utils.round_up_to_12h_interval(
        datetime.strptime(
            dim_kwargs[c.time][c.start_datetime],
            SETTINGS.DATETIME_FORMAT_ISO_8601,
        )
    )
    end_dt = processor_utils.round_down_to_12h_interval(
        datetime.strptime(
            dim_kwargs[c.time][c.end_datetime],
            SETTINGS.DATETIME_FORMAT_ISO_8601,
        )
    )
    # Only include dates after aws_ifs_start_0p25_240h (inclusive)
    available_start = max(
        start_dt,
        datetime.strptime(
            cc.aws_ifs_start_0p25_240h,
            SETTINGS.DATETIME_FORMAT_UTC,
        ),
    )
    times = xr.DataArray(
        pd.date_range(
            available_start,
            end_dt,
            freq="12h",
        ),
        dims=[c.time],
        name=c.time,
    )

    return times


def generate_latitude_coord(bbox: list[float]) -> xr.DataArray:
    return xr.DataArray(
        data=cc.ifs_global_latitudes,
        dims=[c.latitude],
        coords={c.latitude: cc.ifs_global_latitudes},
        name=c.latitude,
        attrs={c.long_name: c.latitude},
    ).sel(latitude=slice(bbox[3], bbox[1]))


def generate_longitude_coord(bbox: list[float]) -> xr.DataArray:
    return xr.DataArray(
        data=cc.ifs_global_longitudes,
        dims=[c.longitude],
        coords={c.longitude: cc.ifs_global_longitudes},
        name=c.longitude,
        attrs={c.long_name: c.longitude},
    ).sel(longitude=slice(bbox[0], bbox[2]))
