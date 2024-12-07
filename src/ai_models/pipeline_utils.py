import xarray as xr
import numpy as np
import os
from glob import glob

from mymetrics import compute_mse, compute_rmse, compute_mae, compute_bias, compute_acc


# SD
class WeatherForecastPipeline:
    def __init__(self, forecast_dir, ground_truth_path, metrics: dict, lead_time=48, time_interval=6):
        self.forecast_dir = forecast_dir
        self.ground_truth_path = ground_truth_path
        self.metrics = metrics
        self.lead_time = lead_time
        self.time_interval = time_interval
        self.models = self._get_models()

    def _get_models(self):
        return [os.path.basename(path) for path in glob(os.path.join(self.forecast_dir, "*")) if os.path.isdir(path)]

    def load_data(self, file_path):
        return xr.open_dataset(file_path, engine="cfgrib")

    def process_model(self, model_name):
        """
        Processes a single model's forecast data.
        
        Args:
            model_name (str): Name of the model.
        
        Returns:
            dict: A dictionary of computed metrics for the model.
        """
        model_dir = os.path.join(self.forecast_dir, model_name)
        forecast_files = sorted(glob(os.path.join(model_dir, "*.grib")))
        ground_truth = self.load_data(self.ground_truth_path)

        results = []

        for forecast_file in forecast_files:
            forecast = self.load_data(forecast_file)
            lead_times = np.arange(0, self.lead_time + 1, self.time_interval)
            
            for lead in lead_times:
                # Extract time slice corresponding to the lead time
                forecast_slice = forecast.sel(time=forecast.time.values[0] + np.timedelta64(lead, 'h'))
                ground_truth_slice = ground_truth.sel(time=forecast_slice.time)

                # Ensure overlapping spatial domain
                common_area = {
                    "latitude": slice(max(forecast_slice.latitude.min(), ground_truth_slice.latitude.min()),
                                      min(forecast_slice.latitude.max(), ground_truth_slice.latitude.max())),
                    "longitude": slice(max(forecast_slice.longitude.min(), ground_truth_slice.longitude.min()),
                                       min(forecast_slice.longitude.max(), ground_truth_slice.longitude.max())),
                }
                forecast_slice = forecast_slice.sel(**common_area)
                ground_truth_slice = ground_truth_slice.sel(**common_area)

                # Compute metrics
                metrics_result = {metric_name: metric_func(forecast_slice.values, ground_truth_slice.values)
                                  for metric_name, metric_func in self.metrics.items()}
                metrics_result["model"] = model_name
                metrics_result["lead_time"] = lead
                results.append(metrics_result)

        return results

    def run(self):
        """
        Runs the evaluation pipeline for all models.
        
        Returns:
            list: A list of dictionaries containing results for all models and lead times.
        """
        all_results = []
        for model in self.models:
            model_results = self.process_model(model)
            all_results.extend(model_results)
        return all_results


# Set paths and parameters
forecast_directory = "/path/to/forecast/models"  # Directory structure: forecast_dir/model_name/*.grib
ground_truth_file = "/path/to/era5/ground_truth.grib"
metrics = {
    "RMSE": compute_rmse,
    "MAE": compute_mae,
    "Bias": compute_bias
}


pipeline = WeatherForecastPipeline(forecast_dir=forecast_directory,
                                   ground_truth_path=ground_truth_file,
                                   metrics=metrics)
results = pipeline.run()


import pandas as pd
results_df = pd.DataFrame(results)


results_df.to_csv("forecast_evaluation_results.csv", index=False)
