# Emissions from Above

Classifying power plant pollution levels via satellite imagery and emissions data.

## Overview
Power plants are among the largest point-source emitters of greenhouse gases,
yet emissions monitoring relies heavily on self-reported data that is delayed,
uneven across jurisdictions, and difficult to verify independently. This project
builds a multimodal ML pipeline that combines satellite observation with
ground-truth emissions records to provide a scalable, independent signal for
environmental monitoring.

## What it does
- **Binary classification** — flags facilities as high- vs. low-pollution using CNNs (ResNet/EfficientNet) on satellite patches centered on plant coordinates
- **Fuel-type detection** — multi-class classification across coal, gas, oil, nuclear, hydro, wind, and solar
- **Emissions regression** — predicts continuous CO₂ output (tons/year, lbs/MWh) from image features fused with structured covariates
- **Anomaly detection** — flags facilities whose visual signature suggests higher emissions than reported

## Data
- **Imagery:** Sentinel-2, Landsat 8/9, accessed via Google Earth Engine
- **Emissions & facility data:** EPA eGRID, EPA CAMPD (hourly CEMS), WRI Global Power Plant Database, EIA Forms 860/923

## Stack
PyTorch · torchvision · HuggingFace · scikit-learn · Google Earth Engine · rasterio · geopandas · PostGIS · MLflow · Grad-CAM

## Team
Rob Scott · Ananya Sen · Mateo Ronquillo · Efrain Rodriguez  
Machine Learning II
