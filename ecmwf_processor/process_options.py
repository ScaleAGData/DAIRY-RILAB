from ecmwf_processor import constants as c

PROCESSOR_KWARGS = {
    c.bands_and_indices_specs: {
        c.ecmwf_ifs: {c.temperature_2m: {c.formula: c.grib2_t2m}},
    },
    c.cube_dimensions: {
        c.time: {
            c.start_datetime: "2024-11-13T00:00:00Z",
            c.end_datetime: "2026-01-01T00:00:00Z",
            # c.end_datetime: "2027-01-01T00:00:00Z",
        },
        c.step: {c.min: 0, c.max: 360},
    },
    c.dst_chunking: {
        c.time: 1,
        c.step: 85,
        c.latitude: 64,
        c.longitude: 64,
        c.level: 2,
        c.soil_layer: 2,
    },
    c.dask_chunking: {
        c.time: 1,
        c.step: -1,
        c.latitude: -1,
        c.longitude: -1,
        c.level: -1,
        c.soil_layer: -1,
    },
    c.delete_everything_before_run: False,
}
