import argparse
import logging
import os
import xarray as xr
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def process_dailydata(ds: xr.Dataset) -> xr.Dataset:
    '''
    Routine to process the data to unify meteorological and flow times,
    and return a new xr.Dataset

    input:
        ds: xr.Dataset of daily data
    output:
        ds: xr.Dataset of daily data with unified times
    '''
    try:
        logger.info(f"Processing xr.Dataset with daily data")
        # drop unnecessary coordinates
        ds = ds.mean(dim="station_number").drop(["lat", "lon"])

        # slices time_array to make dimesion coincide between weather and flow
        # variables
        times_intersect = np.intersect1d(
            ds.time_weather.values, ds.time_flow.values)
        slice_weather = slice(ds.time_weather.values[0], times_intersect[0])
        slice_flow = slice(times_intersect[-1], ds.time_flow.values[-1])

        # auxiliary datasets to be merged; this because time periods are
        # shifted
        ds0 = ds[["temperature", "precipitation"]].sel(
            time_weather=slice_weather).rename({"time_weather": "time"})
        ds1 = ds.sel(time_flow=times_intersect, time_weather=times_intersect)
        ds2 = ds["flow"].sel(time_flow=slice_flow).rename(
            {"time_flow": "time"})

        # Separate ds into dataarrays with same coordinate and merge into
        # dataset
        ds = xr.merge([ds0, ds1["flow"].rename({"time_flow": "time"}), ds1[[
                      "temperature", "precipitation"]].rename({"time_weather": "time"}), ds2])

        return ds
    except Exception as e:
        logger.error(f"There was a problem processing the data: {e}")
        raise e


def daily_to_weekly(ds: xr.Dataset) -> xr.Dataset:
    '''
    Routine to compute weekly averaged data. Anomalies for precipitation and
    flow, and full field for temperatures.

    input:
        ds: xr.Dataset of daily data with unified times
    output:
        ds: xr.Dataset of weekly averaged data
    '''
    try:
        logger.info(f"Calculating weekly variables")

        # Compute weekly mean temperatures
        weekly_temp = ds.temperature.resample(time="1W").mean()

        # Now weekly climatology for water vars
        weekly_avg = ds[["flow", "precipitation"]].assign_coords(
            {"week": ds.time.dt.isocalendar()["week"].values})
        weekly_avg = weekly_avg.groupby("week").mean(
            "time").groupby("week").mean("week")

        # anomaly for water vars
        weekly_water = ds.resample(
            time="1W").mean().groupby("time.week") - weekly_avg

        # Merge temp and water vars
        ds = xr.merge([weekly_water, weekly_temp])

        return ds

    except Exception as e:
        logger.error(f"There was a problem computing weekly variables: {e}")
        raise e


def train_test_dataset(
    weekly_ds: xr.Dataset,
    daily_ds: xr.Dataset,
    lower_frct_time: np.datetime64,
    upper_frct_time: np.datetime64
) -> pd.DataFrame:
    '''
    Outputs a pd.DataFrame of trainig/testing data.

    input:
        weekly_ds: xr.Dataset with weekly averaged data for features
        daily_ds: xr.Dataset with daily flow anomalies for predictand
        lower_frct_time: np.datetime64 with lower limit of prediction period
        upper_frct_time: np.datetime64 with upper limit of prediction period
    output:
        df: pd.DataFrame to be used as train/test dataset
    '''
    try:
        logger.info(f"Preparing data to create train-test dataset")

        # Create slices and time arrays that will be useful to get data
        frct_times_slice = slice(lower_frct_time, upper_frct_time)
        forecast_times = weekly_ds.sel(time=frct_times_slice).time.values
        forecast_times = forecast_times[1:]  # remove first for dflow/dt

        # Get preforecast times to locate first forecast index when using isel
        pre_forecast_times = weekly_ds.isel(
            time=(weekly_ds.time < forecast_times[0])).time.values
        frct_start = len(pre_forecast_times)
        frct_end = frct_start + len(forecast_times)

        # Auxiliary datasets used for readability when filling dataset
        weekly0 = weekly_ds.isel(time=range(frct_start, frct_end))
        weekly_1 = weekly_ds.isel(time=range(frct_start - 1, frct_end - 1))
        weekly_2 = weekly_ds.isel(time=range(frct_start - 2, frct_end - 2))

        # For reference, column names are defined just before filling the dataset.
        # Numbers indicate the time shift from the forecast date, with
        # underscore representing negative shifts (in weeks, except flow7 (predictand),
        # representing a 7 day shift)
        cols = ["flow0",
                "dflow",
                "temp0",
                "temp_1",
                "pr0",
                "pr_1",
                "pr_Autumn",
                "temp_Winter",
                "temp_Spring",
                "flow7"
                ]

        logger.info(f"Filling entries of train-test dataset")

        # Dataset defined first as np.array
        model_data = np.zeros((len(forecast_times), 10))

        # Filling dataset
        model_data[:, 0] = weekly0.flow.values
        model_data[:, 1] = weekly0.flow.values - weekly_1.flow.values
        model_data[:, 2] = weekly0.temperature.values
        model_data[:, 3] = weekly_1.temperature.values
        model_data[:, 4] = weekly0.precipitation.values
        model_data[:, 5] = weekly_1.precipitation.values
        # -999 represents switch off for seasonal features
        model_data[:, 6] = -999
        model_data[:, 7] = -999
        model_data[:, 8] = -999

        # The switch of some variables depends on week of year
        weeks = weekly_ds.week.values

        # Now, iteratively fill entries of seasonal features and predictand
        for i, tm in enumerate(forecast_times):
            week_ind = frct_start + i  # index for isel method
            week = weeks[week_ind]  # get week coresponding to forecast_time tm

            # Seasonal features are filled below. The main criterion is week #,
            # based on the EDA analysis.

            # Seasonal feature: precipitation in Autumn
            if week > 43 and week < 52:
                avg_period = range(week_ind - week + 41, week_ind)
                model_data[i, 6] = weekly_ds.precipitation.isel(
                    time=avg_period).mean("time")

            # Seasonal feature: Temperature in Winter, end of year
            if week > 48:
                avg_period = range(week_ind - week + 46, week_ind)
                model_data[i, 7] = weekly_ds.temperature.isel(
                    time=avg_period).mean("time")
            # Seasonal feature: Temperature in Winter, beginning of year
            if week < 17:
                avg_period = range(week_ind - week - 8, week_ind)
                model_data[i, 7] = weekly_ds.temperature.isel(
                    time=avg_period).mean("time")

            # Seasonal feature: Temperature in Spring
            if week > 18 and week < 24:
                avg_period = range(week_ind - week, week_ind)
                model_data[i, 8] = weekly_ds.temperature.isel(
                    time=avg_period).mean("time")  # tmax spring

            # Now fill predictand. Use "try" because there might be missing
            # dates
            try:
                date_pred = weekly_ds.isel(
                    time=week_ind).time.values + pd.Timedelta("7D")
                model_data[i, 9] = daily_ds.sel(time=date_pred).values
            except BaseException:
                date_pred = weekly_ds.isel(
                    time=week_ind).time.values + pd.Timedelta("8D")
                model_data[i, 9] = daily_ds.sel(time=date_pred).values

        # Now create DataFrame
        df = pd.DataFrame(model_data, index=forecast_times, columns=cols)

        return df

    except Exception as e:
        logger.error(f"There was a problem creating train-test dataset: {e}")
        raise e


def run(args):
    '''
    Function that reads the NetCDF file, calls functions for process the data
    and outputs to a data_file ready for model training

    input:
      args: parser object with file_path and out_path attributes
    output:
      None: the function saves the processed data to out_path
    '''
    
    abs_path = os.path.abspath(args.file_path)
    logger.info(f"Reading file at {abs_path}")
    daily_raw = xr.open_dataset(args.file_path)

    # Get daily data with unified times
    daily_unified = process_dailydata(daily_raw)

    # Get weekly averaged data
    weekly_vars = daily_to_weekly(daily_unified)

    # The 2 following steps are preparations for train/test dataset creation
    # Calculate flow anomalies (to get predictand)
    daily_flow_anomaly = daily_unified.flow.groupby(
        "time.dayofyear") - daily_unified.flow.groupby("time.dayofyear").mean()
    # Get range of forecast dates, used to generate train/test dataset
    times_intersect = np.intersect1d(daily_raw.time_weather.values,
                                     daily_raw.time_flow.values)

    # Get train/test dataset
    train_test_df = train_test_dataset(
        weekly_vars,
        daily_flow_anomaly,
        times_intersect[0],
        times_intersect[-1]
    )

    # Save to csv, to avoid compatibility issues
    # train_test_df.to_pickle(args.out_path)
    train_test_df.to_csv(args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data from NetCDF to get train-test dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to NetCDF file with flow and meteorological data",
        required=True,
    )

    parser.add_argument(
        "--out_path",
        type=str,
        help="Path where train-test dataset will be saved",
        required=True
    )

    args = parser.parse_args()

    run(args)
