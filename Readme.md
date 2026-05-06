# DAIRY-RILAB - ECMWF IFS Data Processor

The Dairy RILAB is a central component of the ScaleAgData project, a European research initiative funded under the Horizon Europe framework (Grant Agreement No. 101083401).

The project aims to improve the environmental sustainability and competitiveness of European agriculture by integrating multi-source data—including satellite imagery (Sentinel-1/2), in-situ sensors, meteorological data, and advanced modeling—into actionable, data-driven solutions. This repository contains the dairy lab component responsible for processing ECMWF IFS data, which serves as the foundation for forecasting milk quality and quantity.

## Usage in the DAIRY RILAB

The ECMWF IFS Data Processor is designed to be used within the DAIRY RILAB's data processing pipeline. It retrieves, processes, and prepares ECMWF IFS data for integration with ERA5 reanalysis data to provide seamsless time series of meteorological variables to be used as covariates in the forecasting models. The processed data is then combined with milk quality and quantity data to train machine learning models that predict milk quality and quantity based on environmental conditions.

## Benefits of the ECMWF IFS Data Processor

- Provision of ECMWF IFS data in cloud-optimized format Zarr for efficient storage and processing within the DAIRY RILAB's data infrastructure.
- Access to historical and near-real-time meteorological forecast data for the DAIRY RILAB to enable near real-time forecastings as well as backtesting of the forecasting models.
