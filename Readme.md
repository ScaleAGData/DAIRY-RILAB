# DAIRY-RILAB - ECMWF IFS Data Processor

The Dairy RILAB is a central component of the ScaleAgData project, a European research initiative funded under the Horizon Europe framework (Grant Agreement No. 101083401).

The project aims to improve the environmental sustainability and competitiveness of European agriculture by integrating multi-source data—including satellite imagery (Sentinel-1/2), in-situ sensors, meteorological data, and advanced modeling—into actionable, data-driven solutions. This repository contains the dairy lab component responsible for processing ECMWF IFS data, which serves as the foundation for forecasting milk quality and quantity.

## Benefits of the ECMWF IFS Data Processor

Processing and Provision of ECMWF IFS data into cloud-optimized format

- Storing ECMWF IFS data in .zarr instead of superseded .grib2 format (as offered by providers) for efficient storage, provision and access.
- All available variables in one cube with flexible chunking
- Access to data from different providers via herbie-data (ECMWF, AWS, ...)

## Usage in the DAIRY RILAB

The ECMWF IFS Data Processor is designed to be used within the DAIRY RILAB's data processing pipeline. It retrieves, processes, and prepares ECMWF IFS data for integration with ERA5 reanalysis data to provide seamsless time series of meteorological variables to be used as covariates in the forecasting models. The processed data is then combined with milk quality and quantity data to train machine learning models that predict milk quality and quantity based on environmental conditions. It enables efficient near real-time forecastings as well as backtesting of the forecasting models.

## Setup Instructions

1. Clone the repository and navigate to the project directory.
2. Run the `setup.sh` script to create a virtual environment and install the required dependencies.
3. Run the `main.py` script to execute the data processing pipeline.

### Configuration

- The processing options can be configured in the `process_options.py` file, where you can specify the variables to be processed, the time range of the final cube, chunking for cloud optimization.
- The `main.py` script can be modified to specify the area of interest (AOI) and the time range for processing.

### Known Issues

- Data access via herbie-data can be interrupted by network issues or API limitations in times where providers upload new data (e.g. 08:00-09:00 UTC). Some basic retry logic is implemented, but it can still lead to failed runs. In this case, simply re-run the `main.py` script. If used in a operational context some retry logic should be implemented in the calling code.
