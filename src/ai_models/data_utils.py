import xarray as xr
import cfgrib



def extract_temp_level500_new(dataset: xr.Dataset, variable: str, pressure_level: str) -> xr.Dataset:
    ds = cfgrib.open_datasets(dataset)
    ds_temp = next(var for var in ds if variable in var.data_vars)[variable]
    ds_temp_level = ds_temp.sel(isobaricInhPa=pressure_level)
    return ds_temp_level