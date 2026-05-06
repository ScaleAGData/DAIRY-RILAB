import numpy as np
from ecmwf_processor import constants as c

###############################################################
###################### SDPP Configs ###########################
###############################################################


AWS_ECMWF_IFS_CONNECTOR_SUBITEM_ROLE = c.connector
AWS_ECMWF_IFS_CONNECTOR_SUBITEM_ID_IN_ROLE = c.single
AWS_ECMWF_IFS_CONNECTOR_ASSET_ID_SUFFIX = c.ecmwf_ifs

DST_SUBITEM_ROLE = c.main
DST_SUBITEM_ID_IN_ROLE = c.subitem
DST_CUBE_ASSET_ID_SUFFIX = c.ecmwf_ifs


###############################################################
################### ECMWF IFS METADATA ########################
###############################################################

aws_ifs_start_0p25_240h = "2024-02-01T00:00:00"
aws_ifs_start_0p25_360h = "2024-11-13T00:00:00"

xcube_dim_order = [
    c.time,
    c.step,
    c.latitude,
    c.longitude,
    c.level,
    c.soil_layer,
]

ifs_global_latitudes = np.linspace(
    90, -90, int((90 - (-90)) / 0.25) + 1, dtype=np.float32
)
ifs_global_longitudes = np.linspace(
    -180, 179.75, int((180 - (-180)) / 0.25), dtype=np.float32
)

ifs_steps360 = [i for i in range(0, 145, 3)] + [i for i in range(150, 361, 6)]

levels_20240201 = [1000, 925, 850, 700, 500, 300, 250, 200, 50]

levels_latest = [
    1000.0,
    925.0,
    850.0,
    700.0,
    600.0,
    500.0,
    400.0,
    300.0,
    250.0,
    200.0,
    150.0,
    100.0,
    50.0,
]

soil_layers = [1, 2, 3, 4]

# see https://codes.ecmwf.int/grib/param-db/ for short name mappings
# regex for db at https://codes.ecmwf.int/grib/param-db/:
# ^(10u|10v|2t|d|gh|lsm|msl|q|r|ro|skt|sp|st|t|tcwv|tp|u|v|vo)$
grib2_ecmwf_xarray_var_mapping_20240201 = {
    "10u": "u10",  # 10 metre U wind component [m s-1]
    "10v": "v10",  # 10 metre V wind component [m s-1]
    "2t": "t2m",  # 2 metre air temperature [K]
    "d": "d",  # Divergence [s-1]
    "gh": "gh",  # Geopotential height [gpm]
    "lsm": "lsm",  # Land-sea mask [(0 - 1)]
    "msl": "msl",  # Mean sea level pressure [Pa]
    "q": "q",  # Specific humidity [kg kg-1]
    "r": "r",  # Relative humidity [%]
    "ro": "ro",  # Water runoff and drainage [kg m-2]
    "skt": "skt",  # Skin temperature [K]
    "sp": "sp",  # Surface pressure [Pa]
    "st": "st",  # Soil temperature [K]
    "t": "t",  # Temperature [K]
    "tcwv": "tcwv",  # Total column vertically-integrated water vapour [kg m-2]
    "tp": "tp",  # Total precipitation [kg m-2]
    "u": "u",  # U wind component [m s-1]
    "v": "v",  # V wind component [m s-1]
    "vo": "vo",  # Vorticity [s-1]
}

# grib2_ecmwf_xarray_var_mapping_latest
# regex for database at https://codes.ecmwf.int/grib/param-db/:
# ^(100u|100v|10fg|10u|10v|2d|asn|d|ewss|gh|lsm|mn2t3|msl|mucape|mx2t3|nsss|ptype|q|r|ro|sdor|sithick|skt|slor|sot|sp|ssr|ssrd|str|strd|sve|svn|t|tcw|tcwv|tp|tprate|ttr|u|v|vo|vsw|w|z|zos)$
grib2_ecmwf_xarray_var_mapping_latest = {
    "100u": "u100",  # 100 metre U wind component [m s-1]
    "100v": "v100",  # 100 metre V wind component [m s-1]
    "10fg": "max_i10fg",  # Maximum 10 metre wind gust since previous post-processing [m s-1]
    "10u": "u10",  # 10 metre U wind component [m s-1]
    "10v": "v10",  # 10 metre V wind component [m s-1]
    "2d": "d2m",  # 2 metre dewpoint temperature [K]
    "asn": "asn",  # Snow albedo [%] or (0 - 1)
    "d": "d",  # Divergence [s-1]
    "ewss": "ewss",  # Time-integrated eastward turbulent surface stress [N m-2 s]
    "gh": "gh",  # Geopotential height [gpm]
    "lsm": "lsm",  # Land-sea mask (0 - 1)
    "mn2t3": "t2m",  # Minimum temperature at 2 metres in the last 3 hours [K]
    "msl": "msl",  # Mean sea level pressure [Pa]
    "mucape": "mucape",  # Most-unstable CAPE [J kg-1]
    "mx2t3": "t2m",  # Maximum temperature at 2 metres in the last 3 hours [K]
    "nsss": "nsss",  # Time-integrated northward turbulent surface stress [N m-2 s]
    "ptype": "ptype",  # Precipitation type (Code table 4.201)
    "q": "q",  # Specific humidity [kg kg-1]
    "r": "r",  # Relative humidity [%]
    "ro": "ro",  # Water runoff and drainage [kg m-2] or [m]
    "sdor": "sdor",  # Standard deviation of sub-gridscale orography [m]
    "sithick": "sithick",  # Sea-ice thickness [m]
    "skt": "skt",  # Skin temperature [K]
    "slor": "slor",  # Slope of sub-gridscale orography [numeric]
    "sot": "sot",  # Soil temperature [K]
    "sp": "sp",  # Surface pressure [Pa] or Wind speed [m s-1]
    "ssr": "ssr",  # Surface net solar radiation [J m-2]
    "ssrd": "ssrd",  # Surface short-wave (solar) radiation downwards [J m-2]
    "str": "str",  # Surface net thermal radiation [J m-2]
    "strd": "strd",  # Surface long-wave (thermal) radiation downwards [J m-2]
    "sve": "sve",  # Eastward surface sea water velocity [m s-1]
    "svn": "svn",  # Northward surface sea water velocity [m s-1]
    "t": "t",  # Temperature [K]
    "tcw": "tcw",  # Total column water [kg m-2]
    "tcwv": "tcwv",  # Total column vertically-integrated water vapour [kg m-2]
    "tp": "tp",  # Total precipitation [kg m-2] or [m]
    "tprate": "tprate",  # Total precipitation rate [kg m-2 s-1] or [m s-1]
    "ttr": "ttr",  # Top net thermal radiation [J m-2]
    "u": "u",  # U component of wind [m s-1]
    "v": "v",  # V component of wind [m s-1]
    "vo": "vo",  # Vorticity (relative) [s-1]
    "vsw": "vsw",  # Volumetric soil moisture [m3 m-3]
    "w": "w",  # Vertical velocity [Pa s-1] or [m s-1]
    "z": "z",  # Geopotential [m2 s-2]
    "zos": "zos",  # Sea surface height [m]
}

grib2_var_dims_20240201 = {
    "10u": ["latitude", "longitude", "step", "time"],
    "10v": ["latitude", "longitude", "step", "time"],
    "2t": ["latitude", "longitude", "step", "time"],
    "d": ["level", "latitude", "longitude", "step", "time"],
    "gh": ["level", "latitude", "longitude", "step", "time"],
    "lsm": ["latitude", "longitude", "step", "time"],
    "msl": ["latitude", "longitude", "step", "time"],
    "q": ["level", "latitude", "longitude", "step", "time"],
    "r": ["level", "latitude", "longitude", "step", "time"],
    "ro": ["latitude", "longitude", "step", "time"],
    "skt": ["latitude", "longitude", "step", "time"],
    "sp": ["latitude", "longitude", "step", "time"],
    "st": ["latitude", "longitude", "step", "time"],
    "t": ["level", "latitude", "longitude", "step", "time"],
    "tcwv": ["latitude", "longitude", "step", "time"],
    "tp": ["latitude", "longitude", "step", "time"],
    "u": ["level", "latitude", "longitude", "step", "time"],
    "v": ["level", "latitude", "longitude", "step", "time"],
    "vo": ["level", "latitude", "longitude", "step", "time"],
}

grib2_var_dims_latest = {
    "100u": ["latitude", "longitude", "step", "time"],
    "100v": ["latitude", "longitude", "step", "time"],
    "10fg": ["latitude", "longitude", "step", "time"],
    "10u": ["latitude", "longitude", "step", "time"],
    "10v": ["latitude", "longitude", "step", "time"],
    "2d": ["latitude", "longitude", "step", "time"],
    "2t": ["latitude", "longitude", "step", "time"],
    "asn": ["latitude", "longitude", "step", "time"],
    "d": ["level", "latitude", "longitude", "step", "time"],
    "ewss": ["latitude", "longitude", "step", "time"],
    "gh": ["level", "latitude", "longitude", "step", "time"],
    "lsm": ["latitude", "longitude", "step", "time"],
    "mn2t3": ["latitude", "longitude", "step", "time"],
    "msl": ["latitude", "longitude", "step", "time"],
    "mucape": ["latitude", "longitude", "step", "time"],
    "mx2t3": ["latitude", "longitude", "step", "time"],
    "nsss": ["latitude", "longitude", "step", "time"],
    "ptype": ["latitude", "longitude", "step", "time"],
    "q": ["level", "latitude", "longitude", "step", "time"],
    "r": ["level", "latitude", "longitude", "step", "time"],
    "ro": ["latitude", "longitude", "step", "time"],
    "sdor": ["latitude", "longitude", "step", "time"],
    "sithick": ["latitude", "longitude", "step", "time"],
    "skt": ["latitude", "longitude", "step", "time"],
    "slor": ["latitude", "longitude", "step", "time"],
    "sot": ["soilLayer", "latitude", "longitude", "step", "time"],
    "sp": ["latitude", "longitude", "step", "time"],
    "ssr": ["latitude", "longitude", "step", "time"],
    "ssrd": ["latitude", "longitude", "step", "time"],
    "str": ["latitude", "longitude", "step", "time"],
    "strd": ["latitude", "longitude", "step", "time"],
    "sve": ["latitude", "longitude", "step", "time"],
    "svn": ["latitude", "longitude", "step", "time"],
    "t": ["level", "latitude", "longitude", "step", "time"],
    "tcw": ["latitude", "longitude", "step", "time"],
    "tcwv": ["latitude", "longitude", "step", "time"],
    "tp": ["latitude", "longitude", "step", "time"],
    "tprate": ["latitude", "longitude", "step", "time"],
    "ttr": ["latitude", "longitude", "step", "time"],
    "u": ["level", "latitude", "longitude", "step", "time"],
    "v": ["level", "latitude", "longitude", "step", "time"],
    "vo": ["level", "latitude", "longitude", "step", "time"],
    "vsw": ["soilLayer", "latitude", "longitude", "step", "time"],
    "w": ["level", "latitude", "longitude", "step", "time"],
    "z": ["latitude", "longitude", "step", "time"],
    "zos": ["latitude", "longitude", "step", "time"],
}


ecmwf_xarray_var_dims_20240201 = {
    "d": ["level", "latitude", "longitude", "step", "time"],
    "gh": ["level", "latitude", "longitude", "step", "time"],
    "lsm": ["latitude", "longitude", "step", "time"],
    "msl": ["latitude", "longitude", "step", "time"],
    "q": ["level", "latitude", "longitude", "step", "time"],
    "r": ["level", "latitude", "longitude", "step", "time"],
    "ro": ["latitude", "longitude", "step", "time"],
    "skt": ["latitude", "longitude", "step", "time"],
    "sp": ["latitude", "longitude", "step", "time"],
    "st": ["latitude", "longitude", "step", "time"],
    "t": ["level", "latitude", "longitude", "step", "time"],
    "t2m": ["latitude", "longitude", "step", "time"],
    "tcwv": ["latitude", "longitude", "step", "time"],
    "tp": ["latitude", "longitude", "step", "time"],
    "u": ["level", "latitude", "longitude", "step", "time"],
    "u10": ["latitude", "longitude", "step", "time"],
    "v": ["level", "latitude", "longitude", "step", "time"],
    "v10": ["latitude", "longitude", "step", "time"],
    "vo": ["level", "latitude", "longitude", "step", "time"],
}

ecmwf_xarray_var_dims_latest = {
    "asn": ["latitude", "longitude", "step", "time"],
    "d": ["level", "latitude", "longitude", "step", "time"],
    "d2m": ["latitude", "longitude", "step", "time"],
    "ewss": ["latitude", "longitude", "step", "time"],
    "gh": ["level", "latitude", "longitude", "step", "time"],
    "lsm": ["latitude", "longitude", "step", "time"],
    "max_i10fg": ["latitude", "longitude", "step", "time"],
    "msl": ["latitude", "longitude", "step", "time"],
    "mucape": ["latitude", "longitude", "step", "time"],
    "nsss": ["latitude", "longitude", "step", "time"],
    "ptype": ["latitude", "longitude", "step", "time"],
    "q": ["level", "latitude", "longitude", "step", "time"],
    "r": ["level", "latitude", "longitude", "step", "time"],
    "ro": ["latitude", "longitude", "step", "time"],
    "sdor": ["latitude", "longitude", "step", "time"],
    "sithick": ["latitude", "longitude", "step", "time"],
    "skt": ["latitude", "longitude", "step", "time"],
    "slor": ["latitude", "longitude", "step", "time"],
    "sot": ["soilLayer", "latitude", "longitude", "step", "time"],
    "sp": ["latitude", "longitude", "step", "time"],
    "ssr": ["latitude", "longitude", "step", "time"],
    "ssrd": ["latitude", "longitude", "step", "time"],
    "str": ["latitude", "longitude", "step", "time"],
    "strd": ["latitude", "longitude", "step", "time"],
    "sve": ["latitude", "longitude", "step", "time"],
    "svn": ["latitude", "longitude", "step", "time"],
    "t": ["level", "latitude", "longitude", "step", "time"],
    "t2m": ["latitude", "longitude", "step", "time"],
    "tcw": ["latitude", "longitude", "step", "time"],
    "tcwv": ["latitude", "longitude", "step", "time"],
    "tp": ["latitude", "longitude", "step", "time"],
    "tprate": ["latitude", "longitude", "step", "time"],
    "ttr": ["latitude", "longitude", "step", "time"],
    "u": ["level", "latitude", "longitude", "step", "time"],
    "u10": ["latitude", "longitude", "step", "time"],
    "u100": ["latitude", "longitude", "step", "time"],
    "v": ["level", "latitude", "longitude", "step", "time"],
    "v10": ["latitude", "longitude", "step", "time"],
    "v100": ["latitude", "longitude", "step", "time"],
    "vo": ["level", "latitude", "longitude", "step", "time"],
    "vsw": ["soilLayer", "latitude", "longitude", "step", "time"],
    "w": ["level", "latitude", "longitude", "step", "time"],
    "z": ["latitude", "longitude", "step", "time"],
    "zos": ["latitude", "longitude", "step", "time"],
}

grib2_var_attrs_20240201 = {
    "10u": {"heightAboveGround": 10.0},
    "10v": {"heightAboveGround": 10.0},
    "2t": {"heightAboveGround": 2.0},
    "d": {},
    "gh": {},
    "lsm": {"surface": 0.0},
    "msl": {"meanSea": 0.0},
    "q": {},
    "r": {},
    "ro": {"surface": 0.0},
    "skt": {"surface": 0.0},
    "sp": {"surface": 0.0},
    "st": {"depthBelowLandLayer": 0.0},
    "t": {},
    "tcwv": {"entireAtmosphere": 0.0},
    "tp": {"surface": 0.0},
    "u": {},
    "v": {},
    "vo": {},
}

grib2_var_attrs_latest = {
    "100u": {"heightAboveGround": 100.0},
    "100v": {"heightAboveGround": 100.0},
    "10fg": {"heightAboveGround": 10.0},
    "10u": {"heightAboveGround": 10.0},
    "10v": {"heightAboveGround": 10.0},
    "2d": {"heightAboveGround": 2.0},
    "2t": {"heightAboveGround": 2.0},
    "asn": {"surface": 0.0},
    "d": {},
    "ewss": {"surface": 0.0},
    "gh": {},
    "lsm": {"surface": 0.0},
    "mn2t3": {"heightAboveGround": 2.0},
    "msl": {"meanSea": 0.0},
    "mucape": {"mostUnstableParcel": 0.0},
    "mx2t3": {"heightAboveGround": 2.0},
    "nsss": {"surface": 0.0},
    "ptype": {"surface": 0.0},
    "q": {},
    "r": {},
    "ro": {"surface": 0.0},
    "sdor": {"surface": 0.0},
    "sithick": {"surface": 0.0},
    "skt": {"surface": 0.0},
    "slor": {"surface": 0.0},
    "sot": {},
    "sp": {"surface": 0.0},
    "ssr": {"surface": 0.0},
    "ssrd": {"surface": 0.0},
    "str": {"surface": 0.0},
    "strd": {"surface": 0.0},
    "sve": {"surface": 0.0},
    "svn": {"surface": 0.0},
    "t": {},
    "tcw": {"entireAtmosphere": 0.0},
    "tcwv": {"entireAtmosphere": 0.0},
    "tp": {"surface": 0.0},
    "tprate": {"surface": 0.0},
    "ttr": {"nominalTop": 0.0},
    "u": {},
    "v": {},
    "vo": {},
    "vsw": {},
    "w": {},
    "z": {"surface": 0.0},
    "zos": {"surface": 0.0},
}

ecmwf_xarray_var_attrs_20240201 = {
    "d": {},
    "gh": {},
    "lsm": {"surface": 0.0},
    "msl": {"meanSea": 0.0},
    "q": {},
    "r": {},
    "ro": {"surface": 0.0},
    "skt": {"surface": 0.0},
    "sp": {"surface": 0.0},
    "st": {"depthBelowLandLayer": 0.0},
    "t": {},
    "t2m": {"heightAboveGround": 2.0},
    "tcwv": {"entireAtmosphere": 0.0},
    "tp": {"surface": 0.0},
    "u": {},
    "u10": {"heightAboveGround": 10.0},
    "v": {},
    "v10": {"heightAboveGround": 10.0},
    "vo": {},
}

ecmwf_xarray_var_attrs_latest = {
    "asn": {"surface": 0.0},
    "d": {},
    "d2m": {"heightAboveGround": 2.0},
    "ewss": {"surface": 0.0},
    "gh": {},
    "lsm": {"surface": 0.0},
    "max_i10fg": {"heightAboveGround": 10.0},
    "msl": {"meanSea": 0.0},
    "mucape": {"mostUnstableParcel": 0.0},
    "nsss": {"surface": 0.0},
    "ptype": {"surface": 0.0},
    "q": {},
    "r": {},
    "ro": {"surface": 0.0},
    "sdor": {"surface": 0.0},
    "sithick": {"surface": 0.0},
    "skt": {"surface": 0.0},
    "slor": {"surface": 0.0},
    "sot": {},
    "sp": {"surface": 0.0},
    "ssr": {"surface": 0.0},
    "ssrd": {"surface": 0.0},
    "str": {"surface": 0.0},
    "strd": {"surface": 0.0},
    "sve": {"surface": 0.0},
    "svn": {"surface": 0.0},
    "t": {},
    "t2m": {"heightAboveGround": 2.0},
    "tcw": {"entireAtmosphere": 0.0},
    "tcwv": {"entireAtmosphere": 0.0},
    "tp": {"surface": 0.0},
    "tprate": {"surface": 0.0},
    "ttr": {"nominalTop": 0.0},
    "u": {},
    "u10": {"heightAboveGround": 10.0},
    "u100": {"heightAboveGround": 100.0},
    "v": {},
    "v10": {"heightAboveGround": 10.0},
    "v100": {"heightAboveGround": 100.0},
    "vo": {},
    "vsw": {},
    "w": {},
    "z": {"surface": 0.0},
    "zos": {"surface": 0.0},
}
