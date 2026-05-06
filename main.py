from pathlib import Path

from ecmwf_processor.processor_register import register_aoi
from ecmwf_processor.process import process

if __name__ == "__main__":

    bbox = [8.25, 53, 10, 54]
    start_datetime = "2026-05-05T00:00:00Z"
    end_datetime = "2026-05-05T00:00:01Z"
    cube, cube_store = register_aoi(bbox=bbox)

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp_dir:
        process(
            bbox=bbox,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            cube=cube,
            cube_store=cube_store,
            workdir=Path(tmp_dir),
        )

    print("a")
