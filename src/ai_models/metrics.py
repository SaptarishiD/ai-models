import numpy as np
import xarray as xr



def mse(forecast: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Compute Mean Squared Error over latitude and longitude."""
    diff = forecast - target
    mse = (diff ** 2).mean(dim=("lat", "lon"))
    return mse

def rmse(forecast: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Compute Root Mean Squared Error over latitude and longitude."""
    return np.sqrt(mse(forecast, target))

def mae(forecast: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Compute Mean Absolute Error over latitude and longitude."""
    diff = forecast - target
    mae = diff.abs().mean(dim=("lat", "lon"))
    return mae

def bias(forecast: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Compute Bias over latitude and longitude."""
    diff = forecast - target
    bias = diff.mean(dim=("lat", "lon"))
    return bias

def acc(forecast: xr.Dataset, target: xr.Dataset, climatology: xr.Dataset) -> xr.Dataset:
    """Compute Anomaly Correlation Coefficient over latitude and longitude."""
    forecast_anom = forecast - climatology
    target_anom = target - climatology
    numerator = (forecast_anom * target_anom).mean(dim=("lat", "lon"))
    denominator = np.sqrt(
        (forecast_anom ** 2).mean(dim=("lat", "lon")) * (target_anom ** 2).mean(dim=("lat", "lon"))
    )
    acc = numerator / denominator
    return acc