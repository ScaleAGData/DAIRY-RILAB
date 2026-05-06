import json
from datetime import datetime, timedelta
import os
from typing import Any, Callable
import numcodecs
import numpy as np
import odc.geo.geobox
import odc.geo.xr  # needed for .odc methods on xarray objects
import pandas as pd
import xarray
import zarr
import zarr.storage

from ecmwf_processor import constants as c
from ecmwf_processor.config import SETTINGS, LOGGER, InternalInvariantError

DEFAULT_COMPUTED_FLAGS_SPECS = {
    c.computed_datetimes: {
        c.dimensions: [c.time],
        c.fill_value: False,
        c.dtype: bool,
        c.attrs: {"_FillValue": False},
    },
    c.last_updated: {
        c.dimensions: [c.time],
        c.fill_value: np.datetime64("NaT", "ns"),
        c.dtype: "datetime64[ns]",
        c.attrs: {"_FillValue": np.datetime64("NaT", "ns")},
    },
}
DEFAULT_COMPUTED_FLAGS_WITH_INDEX_SPECS = {
    c.computed_datetimes: {
        c.dimensions: [c.time, c.index],
        c.fill_value: False,
        c.dtype: bool,
        c.attrs: {"_FillValue": False},
    },
    c.last_updated: {
        c.dimensions: [c.time, c.index],
        c.fill_value: np.datetime64("NaT", "ns"),
        c.dtype: "datetime64[ns]",
        c.attrs: {"_FillValue": np.datetime64("NaT", "ns")},
    },
    c.index_time: {
        c.dimensions: [c.time, c.index],
        c.fill_value: np.datetime64("NaT", "ns"),
        c.dtype: "datetime64[ns]",
        c.attrs: {"_FillValue": np.datetime64("NaT", "ns")},
    },
}


def get_xarray_zarr_encoding(ds: xarray.Dataset) -> dict[str, Any]:
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            # "compressor": None,  # or use e.g. zarr.Blosc(cname="zstd", clevel=5) for compression
            # "compressors": [numcodecs.zarr3.PCodec(cname="zstd", clevel=3, shuffle="shuffle").codec_config]
            # "compressors": [numcodecs.zarr3.PCodec(cname="zstd", clevel=3, shuffle="shuffle").to_dict()]
            # "compressors": [numcodecs.PCodec(cname="zstd", clevel=3, shuffle="shuffle").codec_config]
            "compressors": [numcodecs.Zlib(level=3).get_config()]
            # "compressors": [numcodecs.zarr3.PCodec(cname="zstd", clevel=5).codec_config]
            # "compressors": [numcodecs.zarr3.Blosc(cname="zstd", clevel=5).codec_config]
        }
        # if "_FillValue" in ds[var].attrs:
        #     encoding[var]["_FillValue"] = ds[var].attrs.pop("_FillValue")

    return encoding


def initialize_and_open_zarr_store(
    coords: dict[str, xarray.DataArray],
    lazy_empty_vars_specs: dict[str, dict[str, Any]],
    dst_chunks: dict[str, int],
    dim_order: list[str] | None = None,
    crs: int = None,
    crs_coord_name: str = None,
    user_computed_flags_specs: dict[str, Any] = {},
    cube_attrs={},
    store: zarr.storage.Store | None = None,
    prefix: str | None = None,
    xr_open_zarr_kwargs: dict[str, Any] = {},
    clear_store: bool = True,
) -> tuple[
    xarray.Dataset,
    zarr.storage.Store,
]:
    """
    Initializes  an empty Zarr store with the specified dimensions and
    variables, or opens an existing Zarr store if it is not empty.


    Example usage:
    lazy_solar_irradiance_xr_ds, store = zarr_ops.initialize_empty_zarr_store(
        container_name=DST_CONTAINER_NAME,
        prefix=f"{process_item.id}.zarr",
        dst_chunking=DST_CHUNKS,
        dimensions=coords,
        dim_order=dim_order,
        crs=src_epsg,
        lazy_empty_vars_specs={
            var: {
                c.dimensions: dim_order,
            }
            for var in SOLAR_IRRADIANCE_VARIABLES
        },
    )
    """

    os.makedirs(SETTINGS.DATA_PATH, exist_ok=True)
    store = zarr.DirectoryStore(f"{SETTINGS.DATA_PATH}/{prefix}.zarr")
    is_store_empty: bool = len(store) == 0

    if not is_store_empty and clear_store:
        # clear zarr store
        # rm store recursively
        zarr.storage.rmdir(store)
        is_store_empty: bool = True

    if is_store_empty:
        data_arrays = {}
        initialize_main_cube_variables(
            coords=coords,
            dst_chunking=dst_chunks,
            lazy_empty_vars_specs=lazy_empty_vars_specs,
            data_arrays=data_arrays,
            dim_order=dim_order,
        )
        if dim_order is not None and c.index in dim_order:
            computed_flags: dict[str, Any] = (
                DEFAULT_COMPUTED_FLAGS_WITH_INDEX_SPECS
                | user_computed_flags_specs
            )
            index: xarray.DataArray = coords[c.index]
            index_chunking: int = dst_chunks[c.index]
        else:
            computed_flags: dict[str, Any] = (
                DEFAULT_COMPUTED_FLAGS_SPECS | user_computed_flags_specs
            )
            index: xarray.DataArray | None = None
            index_chunking: int | None = None

        initialize_computed_time_flags_vars(
            data_arrays=data_arrays,
            computed_flags_specs=computed_flags,
            time=coords[c.time],
            time_chunking=dst_chunks[c.time],
            index=index,
            index_chunking=index_chunking,
        )
        lazy_xr_dataset: xarray.Dataset = xarray.Dataset(
            data_arrays,
            coords=coords,
        )
        if crs:
            assign_crs_kwargs: dict[str, Any] = {}
            if crs_coord_name is not None:
                assign_crs_kwargs = {c.crs_coord_name: crs_coord_name}
            lazy_xr_dataset: xarray.Dataset = lazy_xr_dataset.odc.assign_crs(
                crs=crs,
                **assign_crs_kwargs,
            )

        lazy_xr_dataset.attrs = cube_attrs

        create_empty_zarr_cube(
            store=store,
            lazy_empty_dataset=lazy_xr_dataset,
        )

    else:
        LOGGER.info(
            f"Zarr store is not empty. Skipping initialization of empty Zarr store.",
            extra={
                c.event_type: c.EVENT_TYPES.processing_task,
                c.action: "skipping_initialization_of_empty_zarr_store",
            },
        )
        lazy_xr_dataset: xarray.Dataset = xarray.open_zarr(
            store=store,
            chunks=dst_chunks,
            **xr_open_zarr_kwargs,
        )

    return (lazy_xr_dataset, store)


def initialize_main_cube_variables(
    coords: dict[str, xarray.DataArray],
    lazy_empty_vars_specs: dict[str, dict[str, Any]],
    dst_chunking: dict[str, int],
    data_arrays: dict[str, xarray.DataArray],
    dim_order: list[str] | None = None,
    # crs: int | None = None,
) -> None:

    import dask.array

    for var, specs in lazy_empty_vars_specs.items():
        dims = specs[c.dimensions]
        if dim_order is not None:
            dims: list[str] = sorted(dims, key=lambda x: dim_order.index(x))

        dims = tuple(dims)
        data_shape = tuple(len(coords[dim].data) for dim in dims)
        chunking = tuple(dst_chunking[dim] for dim in dims)
        coords_in_var = {dim: coords[dim] for dim in dims}

        dtype = specs.get(c.dtype, c.float64)
        lazy_data = dask.array.empty(data_shape, chunks=chunking, dtype=dtype)

        lazy_xr_dataarray_var = xarray.DataArray(
            lazy_data, coords=coords_in_var, dims=dims
        )
        lazy_xr_dataarray_var.attrs = specs.get(c.attrs, {})
        data_arrays[var] = lazy_xr_dataarray_var


def initialize_computed_time_flags_vars(
    data_arrays: dict[str, xarray.DataArray],
    computed_flags_specs: dict[str, Any],
    time: xarray.DataArray,
    time_chunking: int,
    index: xarray.DataArray | None = None,
    index_chunking: int | None = None,
) -> None:
    """
    Initialize computed flags variables in the Zarr store. These flag
    variables only have the time dimension with possibly the index dim.
    """
    import dask.array

    if index is not None:
        chunks = (time_chunking, index_chunking)
        coords = {c.time: time, c.index: index}
        dims = [c.time, c.index]
        shape = (time.shape[0], index.shape[0])
    else:
        chunks = (time_chunking,)
        coords = {c.time: time}
        dims = [c.time]
        shape = time.shape

    for flag_var, specs in computed_flags_specs.items():
        flag_data_array = xarray.DataArray(
            dask.array.full(
                shape=shape,
                fill_value=specs[c.fill_value],
                dtype=specs[c.dtype],
                # cannot just simply set chunking -1 for full time chunk as
                # different celery workers could write into the same time chunk
                # then e.g. for aws ecmwf preprocessor
                chunks=chunks,
            ),
            coords=coords,
            dims=dims,
        )
        flag_data_array.attrs = specs.get(c.attrs, {})
        data_arrays[flag_var] = flag_data_array


def create_empty_zarr_cube(
    store: zarr.storage.Store,
    lazy_empty_dataset: xarray.Dataset,
) -> None:

    LOGGER.debug(
        f"Creating empty Zarr cube",
        extra={
            c.event_type: c.EVENT_TYPES.processing_task,
            c.action: "creating_empty_zarr_cube",
        },
    )
    lazy_empty_dataset.to_zarr(
        store=store,
        zarr_format=SETTINGS.XCUBE_SUPPORTED_ZARR_FORMAT,
        compute=False,
        safe_chunks=False,
        consolidated=True,
        encoding=get_xarray_zarr_encoding(ds=lazy_empty_dataset),
    )
    LOGGER.info(
        f"Created empty Zarr cube",
        extra={
            c.event_type: c.EVENT_TYPES.processing_task,
            c.action: "created_empty_zarr_cube",
        },
    )


def get_adjusted_discover_queries_time_sel_slice(
    start_datetime: datetime,
    end_datetime: datetime,
    cube_datetimes: np.ndarray,
) -> slice:
    """
    Adjusts the discover queries time selection slice end datetime to ensure
    that the end datetime is not included in the cube datetimes. Therefore, an
    requested time interval [0,2[ can be divided into two intervals [0,1[ and
    [1,2[ such that 1 is not included in both runs.
    """

    casted_cube_datetimes: pd.DatetimeIndex = pd.to_datetime(cube_datetimes)
    if end_datetime in casted_cube_datetimes:
        end_index = casted_cube_datetimes.get_loc(end_datetime) - 1
        if end_index < 0:
            # If the end index is negative, it means that the discover queries
            # end datetime is equal to the start datetime of the cube. In this
            # case, the slice should produce empty results.
            return slice(
                start_datetime - timedelta(hours=1),
                end_datetime - timedelta(hours=1),
            )
        else:
            cube_end_datetime = casted_cube_datetimes[end_index]
            if start_datetime > cube_end_datetime:
                adjusted_run_end_datetime = start_datetime
            adjusted_run_end_datetime = casted_cube_datetimes[end_index]
        # adjusted_run_end_datetime = discover_queries_end_datetime - timedelta(
        #     hours=hours_to_potentially_adjust
        # )
    else:
        adjusted_run_end_datetime = end_datetime

    return slice(start_datetime, adjusted_run_end_datetime)


def determine_run_timestamps_and_slices(
    start_datetime: datetime,
    end_datetime: datetime,
    cube: xarray.Dataset,
    additional_lower_bounds: list[pd.Timestamp] = [],
    additional_upper_bounds: list[pd.Timestamp] = [],
    get_additional_masks_partial_func: Callable = lambda final_cube_sel: [],
    should_compute_full_computed_sel_slice: bool = False,
    should_compute_anyways: bool = False,
    discover_queries_time_sel_slice: slice | None = None,
    should_update_if_result_already_exist: bool = False,
) -> tuple[pd.DatetimeIndex, slice, slice]:

    pre_run_timestamps: pd.DatetimeIndex = determine_run_timestamps(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        final_cube=cube,
        get_additional_masks_partial_func=get_additional_masks_partial_func,
        additional_lower_bounds=additional_lower_bounds,
        additional_upper_bounds=additional_upper_bounds,
        should_compute_anyways=should_compute_anyways,
        discover_queries_time_sel_slice=discover_queries_time_sel_slice,
        should_update_if_result_already_exist=should_update_if_result_already_exist,
    )

    if pre_run_timestamps.empty:
        run_timestamps = pd.DatetimeIndex([])
    elif should_compute_full_computed_sel_slice:
        # Even if their are only a few datetimes to compute, out of
        # simplicity we compute the full range. If this is problematic, this
        # needs to be considered when choosing the discover_opportunity
        # time range.
        run_timestamps = pd.DatetimeIndex(
            cube[c.time]
            .sel(time=slice(pre_run_timestamps[0], pre_run_timestamps[-1]))
            .data
        )
    else:
        run_timestamps = pre_run_timestamps

    time_sel_slice: slice = None
    time_isel_slice: slice = None
    if not run_timestamps.empty:
        time_sel_slice = slice(run_timestamps[0], run_timestamps[-1])
        time_isel_slice = get_time_isel_slice(
            time_sel_slice=time_sel_slice,
            cube=cube,
            # step_size=step_size,
        )

    return (
        run_timestamps,
        time_sel_slice,
        time_isel_slice,
    )


def determine_run_timestamps(
    start_datetime: datetime,
    end_datetime: datetime,
    final_cube: xarray.Dataset,
    should_compute_anyways: bool,
    get_additional_masks_partial_func: Callable = lambda final_cube_sel: [],
    additional_lower_bounds: list[pd.Timestamp] = [],
    additional_upper_bounds: list[pd.Timestamp] = [],
    discover_queries_time_sel_slice: slice | None = None,
    should_update_if_result_already_exist: bool = False,
) -> pd.DatetimeIndex:

    LOGGER.info(
        f"{determine_run_timestamps=}                            \n"
        f"\t triggered from {start_datetime} to {end_datetime}                            \n"
    )

    if discover_queries_time_sel_slice is None:
        discover_queries_time_sel_slice: slice = (
            get_adjusted_discover_queries_time_sel_slice(
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                cube_datetimes=final_cube[c.time].data,
            )
        )
        discover_queries_time_sel_slice = slice(
            discover_queries_time_sel_slice.start.replace(tzinfo=None),
            discover_queries_time_sel_slice.stop.replace(tzinfo=None),
        )
    final_cube_sel: xarray.Dataset = final_cube.sel(
        time=discover_queries_time_sel_slice
    )
    additional_masks: list[np.ndarray] = get_additional_masks_partial_func(
        final_cube_sel,
    )
    lower_bound: pd.Timestamp = max(
        [discover_queries_time_sel_slice.start] + additional_lower_bounds,
    )
    upper_bound: pd.Timestamp = min(
        [discover_queries_time_sel_slice.stop] + additional_upper_bounds,
    )

    full_cube_datetimes = pd.DatetimeIndex(final_cube_sel[c.time].data)
    should_ignore_computed_datetimes_flags = (
        should_update_if_result_already_exist
    )

    if should_ignore_computed_datetimes_flags or should_compute_anyways:
        full_cube_datetimes_to_compute: pd.DatetimeIndex = full_cube_datetimes

    else:
        full_cube_computed_datetimes: np.ndarray = final_cube_sel[
            c.computed_datetimes
        ].data
        mask_not_computed = ~full_cube_computed_datetimes
        final_mask: np.ndarray = mask_not_computed

        for additional_mask in additional_masks:
            final_mask &= additional_mask

        full_cube_datetimes_to_compute: pd.DatetimeIndex = full_cube_datetimes[
            final_mask
        ]

    run_timestamps: pd.DatetimeIndex = full_cube_datetimes_to_compute[
        (lower_bound <= full_cube_datetimes_to_compute)
        & (full_cube_datetimes_to_compute <= upper_bound)
    ]

    return run_timestamps


def get_time_isel_slice(
    time_sel_slice: slice,
    cube: xarray.Dataset,
) -> slice:

    time_index = cube[c.time].to_index()
    start_idx = time_index.get_indexer([pd.Timestamp(time_sel_slice.start)])[0]
    stop_idx = time_index.get_indexer([pd.Timestamp(time_sel_slice.stop)])[0]
    return slice(start_idx, stop_idx + 1)


def write_timestamp_flags(
    dst_cube: xarray.Dataset,
    dst_cube_store: "zarr.storage.Store",
    run_timestamps: pd.DatetimeIndex,
    time_sel_slice: slice,
    time_isel_slice: slice,
    user_timestamp_flag_vars_to_funcs: dict[str, Callable | str] = {},
) -> None:

    default_timestamp_flag_vars_to_funcs: dict[str, Callable] = {
        c.computed_datetimes: set_run_timestamp_flags,
        c.last_updated: set_last_updated_flag_part_func,
    }
    # user will overwrite default
    timestamp_flag_vars_to_funcs: dict[str, Callable] = (
        get_timestamp_flag_vars_to_funcs(
            user_timestamp_flag_vars_to_funcs=user_timestamp_flag_vars_to_funcs,
            default_timestamp_flag_vars_to_funcs=default_timestamp_flag_vars_to_funcs,
        )
    )
    region = {c.time: time_isel_slice}

    for (
        flag_var,
        set_flag_var_func,
    ) in timestamp_flag_vars_to_funcs.items():
        run_timestamp_flags: xarray.DataArray = (
            dst_cube[flag_var]
            .sel(time=time_sel_slice)
            .load()
            .drop_attrs()
            .drop_vars(c.crs, errors="ignore")
        )
        mask_run_timestamps: np.ndarray = np.isin(
            run_timestamp_flags[c.time].values, run_timestamps
        )
        set_flag_var_func(
            run_timestamp_flags=run_timestamp_flags,
            mask_run_timestamps=mask_run_timestamps,
        )
        run_timestamp_flags_ds = xarray.Dataset(
            {flag_var: run_timestamp_flags}
        )
        run_timestamp_flags_ds = run_timestamp_flags_ds.drop_vars(
            [
                coord
                for coord in run_timestamp_flags_ds.coords
                if coord != c.time
            ]
        )
        run_timestamp_flags_ds.to_zarr(
            store=dst_cube_store,
            # compute=True,
            region=region,
            safe_chunks=False,
        )


def set_run_timestamp_flags(
    run_timestamp_flags: xarray.DataArray,
    mask_run_timestamps: np.ndarray,
    lower_bounds: list[pd.Timestamp] = [],
    upper_bounds: list[pd.Timestamp] = [],
) -> None:

    lower_bound: pd.Timestamp = max(
        lower_bounds,
        default=run_timestamp_flags[c.time].values.min(),
    )
    upper_bound: pd.Timestamp = min(
        upper_bounds,
        default=run_timestamp_flags[c.time].values.max(),
    )

    mask: np.ndarray = (
        mask_run_timestamps
        & (lower_bound <= run_timestamp_flags[c.time].values)
        & (run_timestamp_flags[c.time].values <= upper_bound)
    )

    run_timestamp_flags.values[mask] = True


def set_last_updated_flag_part_func(
    run_timestamp_flags: xarray.DataArray,
    mask_run_timestamps: np.ndarray,
) -> None:
    """Fix timestamp to utc now"""
    set_timestamp(
        run_timestamp_flags=run_timestamp_flags,
        mask_run_timestamps=mask_run_timestamps,
        timestamp=pd.Timestamp.now(tz=c.UTC),
    )


def set_timestamp(
    run_timestamp_flags: xarray.DataArray,
    mask_run_timestamps: np.ndarray,
    timestamp: pd.Timestamp,
) -> None:
    run_timestamp_flags[mask_run_timestamps] = timestamp


def get_timestamp_flag_vars_to_funcs(
    user_timestamp_flag_vars_to_funcs: dict[str, Callable | str],
    default_timestamp_flag_vars_to_funcs: dict[str, Callable],
) -> dict[str, Callable]:
    timestamp_flag_vars_to_funcs: dict[str, Callable] = {
        var: func for var, func in default_timestamp_flag_vars_to_funcs.items()
    }
    for (
        flag_var,
        set_flag_var_func,
    ) in user_timestamp_flag_vars_to_funcs.items():
        timestamp_flag_vars_to_funcs[flag_var] = set_flag_var_func

    return timestamp_flag_vars_to_funcs


def initialize_dst_store(
    coords: dict[str, xarray.DataArray],
    dim_order: list[str],
    src_var_dims: dict[str, Any],
    src_var_attrs: dict[str, Any],
    src_to_final_renaming: dict[str, str],
    main_vars_to_specs: dict[str, dict],
    dst_chunks: dict[str, Any],
    prefix: str,
    clear_store_before_run: bool,
    computed_flags_specs: dict[str, dict] = {},
    xr_open_zarr_kwargs: dict[str, Any] = {},
) -> tuple[xarray.Dataset, zarr.storage.Store]:
    """
    Initializes the final cube asset and its backing store if they do not
    already exist, or clears them if configured to do so. This ensures the
    store is ready before any processing begins, avoiding the need for
    coordination between multiple workers that might otherwise attempt to
    initialize the store concurrently.
    """

    LOGGER.info(
        "Initializing destination Zarr store",
        extra={
            c.action: "initializing_destination_zarr_store",
        },
    )

    cube_attrs = {
        c.cube_time_start: pd.to_datetime(coords[c.time][0].data).strftime(
            SETTINGS.DATETIME_FORMAT_UTC
        ),
        c.last_processed: datetime.now().strftime(
            SETTINGS.DATETIME_FORMAT_UTC
        ),
        c.cube_time_end: pd.to_datetime(coords[c.time][-1].data).strftime(
            SETTINGS.DATETIME_FORMAT_UTC
        ),
    }
    main_vars_specs: dict[str, dict[str, Any]] = get_main_vars_specs(
        src_to_final_renaming=src_to_final_renaming,
        main_vars_to_specs=main_vars_to_specs,
        src_vars_dims=src_var_dims,
        src_vars_attrs=src_var_attrs,
        coords=coords,
    )
    cube, cube_store, *_ = initialize_and_open_zarr_store(
        prefix=prefix,
        coords=coords,
        dim_order=dim_order,
        lazy_empty_vars_specs=main_vars_specs,
        user_computed_flags_specs=computed_flags_specs,
        dst_chunks=dst_chunks,
        crs=4326,
        crs_coord_name=c.crs,
        cube_attrs=cube_attrs,
        xr_open_zarr_kwargs=xr_open_zarr_kwargs,
        clear_store=clear_store_before_run,
    )

    return cube, cube_store


def get_main_vars_specs(
    src_to_final_renaming: dict[str, str],
    main_vars_to_specs: dict[str, dict[str, Any]],
    src_vars_dims: dict[str, list[float]],
    src_vars_attrs: dict[str, dict[str, Any]],
    coords: dict[str, xarray.DataArray],
) -> dict[str, dict[str, Any]]:

    final_main_vars_specs = {}
    for src_var, final_var in src_to_final_renaming.items():
        src_var_dims: list[str] = json.loads(
            json.dumps(src_vars_dims[src_var])
        )
        for dim, coord in coords.items():
            if (
                dim in src_var_dims
                and coord.size == 1
                and main_vars_to_specs[final_var]
                .get(c.cube_dimensions, {})
                .get(dim, {})
                .get(c.squeeze, False)
            ):
                src_var_dims.remove(dim)
        final_main_vars_specs[final_var] = {
            c.dimensions: src_var_dims,
            c.attrs: src_vars_attrs[src_var],
        }

    return final_main_vars_specs
