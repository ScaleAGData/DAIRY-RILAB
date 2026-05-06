import socket
import time
from datetime import UTC, datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import urllib3
import xarray as xr
import zarr
import zarr.storage
from herbie import Herbie

from ecmwf_processor import constants as c
from ecmwf_processor import geometry, zarr_ops
from ecmwf_processor.config import LOGGER, SETTINGS
from ecmwf_processor import processor_constants as cc
from ecmwf_processor import processor_utils
from ecmwf_processor.process_options import PROCESSOR_KWARGS

STEP_SIZE: int = 12 * int(c.factor_ns_to_h)


def process(
    bbox: list[float],
    start_datetime: str,
    end_datetime: str,
    cube: xr.Dataset,
    cube_store: zarr.storage.Store,
    workdir: Path,
) -> None:

    # AWS resources: https://aws.amazon.com/marketplace/pp/prodview-3ibagms7ky4ec
    # 20240201 first ifs with 0p25 and 240h forecast
    # 20241113 first ifs with 0p25 and 360h forecast

    LOGGER.info(f"{process.__name__=} Start to process AWS ECMWF IFS data...")

    (
        run_timestamps,
        time_sel_slice,
        time_isel_slice,
    ) = zarr_ops.determine_run_timestamps_and_slices(
        start_datetime=pd.Timestamp(start_datetime),
        end_datetime=pd.Timestamp(end_datetime),
        cube=cube,
        additional_lower_bounds=[
            datetime.strptime(
                cc.aws_ifs_start_0p25_360h,
                SETTINGS.DATETIME_FORMAT_UTC,
            ),
        ],
        additional_upper_bounds=[latest_dt_via_herbie(workdir=workdir)],
    )
    grib2_to_final_renaming: dict[str, str] = (
        processor_utils.get_grib2_to_final_renaming(PROCESSOR_KWARGS)
    )
    bbox_final: list[float] = geometry.compute_ecmwf_era5_adjusted_bbox(
        bbox=bbox,
        bbox_px_padding=PROCESSOR_KWARGS.get(c.bbox_px_padding, 0),
    )

    LOGGER.info(f"{len(run_timestamps)=}")

    for run_timestamp in run_timestamps:
        download_buildzarr_for_all_vars_and_one_timestamp(
            bbox_final=bbox_final,
            run_timestamp=run_timestamp,
            step_kwargs=PROCESSOR_KWARGS[c.cube_dimensions][c.step],
            final_cube=cube,
            grib2_to_final_renaming=grib2_to_final_renaming,
            workdir=workdir,
            final_cube_store=cube_store,
            cube_dimensions=PROCESSOR_KWARGS[c.cube_dimensions],
        )
    if not run_timestamps.empty:
        zarr_ops.write_timestamp_flags(
            dst_cube_store=cube_store,
            dst_cube=cube,
            run_timestamps=run_timestamps,
            time_sel_slice=time_sel_slice,
            time_isel_slice=time_isel_slice,
        )


def latest_dt_via_herbie(
    workdir: Path,
) -> datetime:
    now = datetime.now(UTC)
    if now.hour > 12:
        latest = now.replace(hour=12, minute=0, second=0, microsecond=0)
    else:
        latest = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for hours_ago in range(0, 48 + 1, 12):  # IFS runs every 12 hours
        check_dt = latest - timedelta(hours=hours_ago)
        H = get_herbie(
            datetime=check_dt,
            step=0,
            workdir=workdir,
        )
        if H.grib is not None:
            return H.date.to_pydatetime()
    return None


def download_buildzarr_for_all_vars_and_one_timestamp(
    bbox_final: list[int],
    run_timestamp: pd.Timestamp,
    step_kwargs: dict[str, Any],
    final_cube: xr.Dataset,
    grib2_to_final_renaming: dict[str, str],
    workdir: Path,
    final_cube_store: "zarr.storage.Store",
    cube_dimensions: dict[str, Any],
) -> None:

    for grib2_var, final_var in grib2_to_final_renaming.items():
        LOGGER.info(
            f"{download_buildzarr_for_all_vars_and_one_timestamp=} "
            f"Download Data and Upload Temporal Zarr for {run_timestamp}/{final_var}"
        )
        download_buildzarr_for_one_var_and_timestamp(
            bbox_final=bbox_final,
            run_timestamp=run_timestamp,
            grib2_var=grib2_var,
            step_kwargs=step_kwargs,
            final_cube=final_cube,
            grib2_to_final_renaming=grib2_to_final_renaming,
            workdir=workdir,
            final_cube_store=final_cube_store,
            cube_dimensions=cube_dimensions,
        )


def download_buildzarr_for_one_var_and_timestamp(
    bbox_final: list[int],
    run_timestamp: pd.Timestamp,
    grib2_var: str,
    step_kwargs: dict[str, Any],
    final_cube: xr.Dataset,
    grib2_to_final_renaming: dict[str, str],
    workdir: Path,
    final_cube_store: "zarr.storage.Store",
    cube_dimensions: dict[str, Any],
) -> None:

    steps: list[int] = determine_available_steps_for_run(
        run_timestamp=run_timestamp,
        step_kwargs=step_kwargs,
    )

    step_datasets = []
    dst_var = grib2_to_final_renaming[grib2_var]
    for step in steps:
        step_ds: xr.Dataset = download_and_build_ds_for_single_step(
            final_cube_coords={
                dim: final_cube.coords[dim] for dim in final_cube.dims
            },
            final_cube_var=final_cube[dst_var],
            run_timestamp=run_timestamp,
            step=step,
            grib2_var=grib2_var,
            final_name=dst_var,
            lat_slice=slice(bbox_final[3], bbox_final[1]),
            lon_slice=slice(bbox_final[0], bbox_final[2]),
            workdir=workdir,
            # dst_var_dims=cube_dimensions,
        )
        step_datasets.append(step_ds)

    var_dataset: xr.Dataset = xr.concat(step_datasets, dim=c.step)

    cube_time_start = pd.Timestamp(final_cube[c.time].data[0])
    run_timestamp_idx = int(
        (run_timestamp - cube_time_start).value / STEP_SIZE
    )
    region = {
        c.time: slice(run_timestamp_idx, run_timestamp_idx + 1),
        c.step: slice(0, var_dataset.coords[c.step].size),
        c.latitude: slice(0, len(var_dataset.coords[c.latitude])),
        c.longitude: slice(0, len(var_dataset.coords[c.longitude])),
    }
    if c.level in var_dataset.dims:
        region[c.level] = slice(0, len(var_dataset.coords[c.level]))
    if c.soil_layer in var_dataset.dims:
        region[c.soil_layer] = slice(0, len(var_dataset.coords[c.soil_layer]))

    var_dataset.to_zarr(
        store=final_cube_store,
        # compute=False,
        region=region,
        safe_chunks=False,
    )


def determine_available_steps_for_run(
    run_timestamp: pd.Timestamp, step_kwargs: dict
) -> list[int]:

    steps_360h_fc = list(chain(range(0, 145, 3), range(150, 361, 6)))

    steps = [
        step
        for step in steps_360h_fc
        if step_kwargs[c.min] <= step <= step_kwargs[c.max]
    ]

    if run_timestamp < datetime.strptime(
        cc.aws_ifs_start_0p25_360h,
        SETTINGS.DATETIME_FORMAT_UTC,
    ):
        steps = [step for step in steps if step <= 240]

    return steps


def download_and_build_ds_for_single_step(
    run_timestamp: pd.Timestamp,
    step: int,
    grib2_var: str,
    final_cube_coords: dict[str, xr.DataArray],
    final_cube_var: xr.DataArray,
    final_name: str,
    lat_slice: slice,
    lon_slice: slice,
    workdir: Path,
) -> xr.Dataset:

    dst_dims = set(final_cube_coords.keys())
    global_grib_var_dims = {
        **cc.grib2_var_dims_20240201,
        **cc.grib2_var_dims_latest,
    }

    if (c.level in dst_dims or c.soil_layer in dst_dims) and (
        c.level in global_grib_var_dims[grib2_var]
        or c.soil_layer in global_grib_var_dims[grib2_var]
    ):
        # If there is a level-like dimension, we need to get the values for it
        # from the final_cube to construct the search string.
        # Otherwise, we can just use the grib2_var name.
        if c.level in global_grib_var_dims[grib2_var]:
            dst_level_like_dim: str = c.level
        elif c.soil_layer in global_grib_var_dims[grib2_var]:
            dst_level_like_dim: str = c.soil_layer
        else:
            raise ValueError(
                f"Neither {c.level} nor {c.soil_layer} found in global_grib_var_dims for {grib2_var}"
            )
        dst_dim_values: np.ndarray = final_cube_coords[
            dst_level_like_dim
        ].values
        if dst_level_like_dim == c.level:
            search_strs = [f":{grib2_var}:{val}" for val in dst_dim_values]

        elif dst_level_like_dim == c.soil_layer:
            search_strs = [f":{grib2_var}l{val}:" for val in dst_dim_values]

    else:
        search_strs = [f":{grib2_var}:"]
        dst_level_like_dim = None

    H: Herbie = get_herbie(
        datetime=run_timestamp,
        step=step,
        workdir=workdir,
    )
    step_datasets = []
    for search in search_strs:
        retry_idx = 0
        while retry_idx < SETTINGS.HERBIE_MAX_RETRIES:
            try:
                pre_ds: xr.Dataset = (
                    H.xarray(search=search, verbose=False)
                    .sel(latitude=lat_slice, longitude=lon_slice)
                    .drop_attrs()
                )
                break
            except Exception as exception:
                LOGGER.warning(
                    f"Attempt {retry_idx}/{SETTINGS.HERBIE_MAX_RETRIES} failed to download {search} for run {run_timestamp} and step {step}: {exception}"
                )
                retry_idx += 1
                if retry_idx >= SETTINGS.HERBIE_MAX_RETRIES:
                    raise exception
            time.sleep(1)

        if c.isobaricinhpa in list(pre_ds.coords):
            ds: xr.Dataset = pre_ds.rename({c.isobaricinhpa: c.level})
        else:
            ds: xr.Dataset = pre_ds
        step_datasets.append(ds)

    if dst_level_like_dim is not None:
        step_ds: xr.Dataset = xr.concat(step_datasets, dim=dst_level_like_dim)
        if len(dst_dim_values) < 2 and dst_level_like_dim not in step_ds.dims:
            step_ds = step_ds.expand_dims(dst_level_like_dim)

    else:
        step_ds = xr.merge(step_datasets)

    coords_to_drop = [
        coord for coord in step_ds.coords if coord not in final_cube_var.dims
    ]
    dims_to_drop = [
        dim for dim in step_ds.dims if dim not in final_cube_var.dims
    ]
    if dims_to_drop:
        step_ds = step_ds.squeeze(dim=dims_to_drop, drop=True)

    step_ds = (
        step_ds.drop_vars(coords_to_drop, errors="ignore")
        .expand_dims(c.time)
        .rename_vars(
            {cc.grib2_ecmwf_xarray_var_mapping_20240201[grib2_var]: final_name}
        )
    )

    return step_ds


def get_herbie(
    datetime: datetime,
    step: int,
    workdir: Path,
) -> Herbie:

    RETRIES = 10
    WAIT = 0.05

    for attempt in range(RETRIES):
        try:
            # Ensure datetime is timezone-naive in UTC
            if hasattr(datetime, "tzinfo") and datetime.tzinfo is not None:
                naive_datetime = datetime.replace(tzinfo=None)
            else:
                naive_datetime = datetime
            herbie = Herbie(
                date=naive_datetime,
                model=c.ifs,
                priority=c.aws,
                product=c.oper,
                fxx=step,
                verbose=False,
                save_dir=workdir.as_posix(),
                overwrite=True,
            )
            break
        except (
            urllib3.exceptions.ProtocolError,
            requests.exceptions.ConnectionError,
            ConnectionResetError,
            socket.error,
        ) as e:
            if attempt < RETRIES - 1:
                time.sleep(WAIT)
            else:
                LOGGER.error(
                    f"Failed to initialize Herbie for datetime={datetime}, step={step} after {RETRIES} attempts: {e}"
                )
                raise RuntimeError(
                    f"Herbie initialization failed for datetime={datetime}, step={step}"
                ) from e
    return herbie
