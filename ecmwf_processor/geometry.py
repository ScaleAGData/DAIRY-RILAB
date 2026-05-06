import math


def compute_ecmwf_era5_adjusted_bbox(
    bbox: list[float], bbox_px_padding: int
) -> list[float]:
    """
    Adjust bbox to get a bbox that is aligned to the 0.25 degree grid with
    (0,0) anchor point. Otherwise, e.g. slice(0.1, 0.2) would not return the
    pixel at (0, 0) in .sel. This will transform this slice to slice(0,
    0.00001), covering the pixel at (0, 0) in .sel.

    UPDATE: Added 0.25 so that bbox has at least 2 pixels in each dimension.
    Xcube seems not to handle 1 pixel in each dimension correctly.
    """
    resolution = 0.25

    def lower(coord: float) -> float:
        return math.floor(coord / resolution) * resolution

    def upper(coord: float) -> float:
        return math.ceil(coord / resolution) * resolution

    floored_bbox: list[float] = [
        lower(bbox[0]),
        upper(bbox[1]),
        upper(bbox[2]),
        lower(bbox[3]),
    ]

    if bbox_px_padding:
        padding: float = bbox_px_padding * 0.25
        final_bbox: list[float] = [
            floored_bbox[0] - padding,
            floored_bbox[1] - padding,
            floored_bbox[2] + padding,
            floored_bbox[3] + padding,
        ]
    else:
        final_bbox: list[float] = floored_bbox

    return final_bbox
