import xarray as xr
import cfgrib



def extract_temp_pressure_level(dataset: xr.Dataset, variable: str, pressure_level: str) -> xr.Dataset:
    ds = cfgrib.open_datasets(dataset)
    ds_temp = next(var for var in ds if variable in var.data_vars)[variable]
    ds_temp_level = ds_temp.sel(isobaricInhPa=pressure_level)
    return ds_temp_level


def process_and_combine(initial_df, model, lead_time, model_dir):
    final_df = initial_df
    if model == 'pangu':
        for i in range(6, lead_time, 6):
            pangu_df_temp = xr.merge(cfgrib.open_datasets(f"{model_dir}/out-pangu-{i}.grib"), compat="override")
            print(pangu_df_temp)
            final_df = xr.concat([initial_df, pangu_df_temp], dim="step")
    del pangu_df_temp
    return initial_df

