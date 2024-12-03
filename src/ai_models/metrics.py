import numpy as np
import xarray as xr



## adapting from weatherbench:
def rmse(forecast: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    diff = forecast - target
    diff = (diff**2)
    diff = diff.mean(("lat", "lon"))
    return np.sqrt(diff)