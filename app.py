import pandas as pd
import pandas_access as mdb
import itertools

import numpy as np
import fbprophet as ph
from fbprophet.diagnostics import cross_validation, performance_metrics

import matplotlib.pyplot as plt
import datetime
from time import process_time
import time

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

# from scipy.stats import boxcox
# from scipy.special import inv_boxcox

# https://medium.com/mlearning-ai/forecast-using-prophet-canadian-natural-gas-production-dataset-b1f9c57548d8
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import streamlit as st  # pylint: disable=import-error

import pytz

eastern = pytz.timezone("US/Eastern")

import pathlib
from apps.utils import predict_the_future, site_specific_settings

import sqlite3
from sqlite3 import Error

today_date = "today"
# today_date = "5-24-2021"

##
## IDEAS TO IMPROVE FORECASTING
##

## Asphalt plants - 15 days of use, higher CPS (.5)
## Load usage for previous year(s) same month

# Configuration Stuff
forecast_for = (pd.to_datetime(today_date) + datetime.timedelta(1)).normalize()
days_to_forecast = 1

days_to_forecast = 1
if pd.to_datetime(today_date).weekday() == 4:
    days_to_forecast = 3  # Weekend!

st.set_page_config(page_title="ENG Prophet Forecasting", layout="wide")

#
# Hyperparameter Tuning
#
############################################################
param_grid = {
    # "changepoint_prior_scale": [0.01, 0.5, 0.9],
    "changepoint_prior_scale": [0.001, 0.01, 0.5, 0.9],
    # "changepoint_range": [0.05, 0.10, 0.25, 0.5, 0.75],
    # "changepoint_range": [0.8],
    "seasonality_mode": ["additive", "multiplicative"],
    "seasonality_prior_scale": [0.01, 5, 10.0],
    # "daily_seasonality": [True, False],
    # "weekly_seasonality": [True, False],
}

# Generate all combinations of parameters
all_params = [
    dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
]
rmses = []  # Store the RMSEs for each params here

############################################################

# Data retrieval method  (PYODBC, PANDAS_ACCESS)
data_retrieval_method = "PANDAS_ACCESS"

root_share = "/mnt/arminius"

MDB = root_share + "/DailyBalancing_be.mdb"
DRV = "{Microsoft Access Driver (*.mdb)}"
USR = ""
PWD = ""

# SQLite database for storing forecasting information
forecasting_db_location = "file:" + root_share + "/Backend/Forecast_Data.db?mode=ro"

#
# Areas lookup
#

if data_retrieval_method == "PANDAS_ACCESS":

    # Get data from access database
    categories_df = mdb.read_table(MDB, "tblUseCategory")

    categories_values = categories_df["Description"].tolist()
    categories_values.insert(0, "All Categories")
    categories_options = categories_df["CategoryID"].tolist()
    categories_options.insert(0, 0)
    categories_dic = dict(zip(categories_options, categories_values))

    areas_df = mdb.read_table(MDB, "tblArea")

    # Remove areas that we aren't including in the forecast
    areas_df = areas_df[(areas_df.IncludeOnReport == 1)]
    areas_df.sort_values(by=["Description"], inplace=True)

    # Remove columns that we don't care about
    areas_df = areas_df[["AreaID", "Description", "WeatherStationID"]].dropna()

    areas_values = areas_df["Description"].tolist()
    areas_values.insert(0, "All Areas")
    areas_options = areas_df["AreaID"].tolist()
    areas_options.insert(0, 0)
    areas_dic = dict(zip(areas_options, areas_values))


def main():

    st.sidebar.title("Action")
    app_mode = st.sidebar.selectbox(
        "Choose Action",
        ["Run Site Forecast", "Run Full Forecast", "Forecasts", "Customer Details"],
    )

    selected_area = st.sidebar.selectbox(
        "Select an area", areas_options, format_func=lambda x: areas_dic[x]
    )

    selected_category = st.sidebar.selectbox(
        "Select a category", categories_options, format_func=lambda x: categories_dic[x]
    )

    days_of_usage = st.sidebar.selectbox(
        "Days of usage", (7, 15, 21, 30, 45, 60, 90, 120, 180, 365, 730, 1095), index=3
    )

    category_specific_hyperparameters = st.sidebar.checkbox(
        "Category Specific Hyperparameters", value=False
    )

    site_specific_hyperparameters = st.sidebar.checkbox(
        "Site Specific Hyperparameters",
        value=False,
        help="Pulls site settings entered into SQLite database.  These values will override any settings selected on this page.",
    )

    changepoint_prior_scale = st.sidebar.text_input(
        "Changepoint Prior Scale",
        value=0.05,
        help="This is probably the most impactful parameter. It determines the flexibility of the "
        + "trend, and in particular how much the trend changes at the trend changepoints. As "
        + "described in this documentation, if it is too small, the trend will be underfit and "
        + "variance that should have been modeled with trend changes will instead end up being "
        + "handled with the noise term. If it is too large, the trend will overfit and in the "
        + "most extreme case you can end up with the trend capturing yearly seasonality. The "
        + "default of 0.05 works for many time series, but this could be tuned; a range of "
        + "[0.001, 0.5] would likely be about right. Parameters like this (regularization "
        + "penalties; this is effectively a lasso penalty) are often tuned on a log scale. ",
    )

    changepoint_range = st.sidebar.text_input(
        "Changepoint Range",
        value=0.8,
        help="This is the proportion of the history in which the trend is allowed to change. "
        + "This defaults to 0.8, 80% of the history, meaning the model will not fit any trend "
        + "changes in the last 20% of the time series. This is fairly conservative, to avoid "
        + "overfitting to trend changes at the very end of the time series where there isn’t "
        + "enough runway left to fit it well. With a human in the loop, this is something "
        + "that can be identified pretty easily visually: one can pretty clearly see if the "
        + "forecast is doing a bad job in the last 20%. In a fully-automated setting, it may "
        + "be beneficial to be less conservative. It likely will not be possible to tune this "
        + "parameter effectively with cross validation over cutoffs as described above. The "
        + "ability of the model to generalize from a trend change in the last 10% of the time "
        + "series will be hard to learn from looking at earlier cutoffs that may not have trend "
        + "changes in the last 10%. So, this parameter is probably better not tuned, except "
        + "perhaps over a large number of time series. In that setting, [0.8, 0.95] may be "
        + "a reasonable range.",
    )

    seasonality_mode = st.sidebar.selectbox(
        "Seasonality Mode", ("Additive", "Multiplicative")
    )

    seasonality_prior_scale = st.sidebar.text_input(
        "Seasonality Prior Scale",
        value=10,
        help="This parameter controls the flexibility of the seasonality. Similarly, a large value "
        + "allows the seasonality to fit large fluctuations, a small value shrinks the magnitude "
        + "of the seasonality. The default is 10., which applies basically no regularization. "
        + "That is because we very rarely see overfitting here (there’s inherent regularization "
        + "with the fact that it is being modeled with a truncated Fourier series, so it’s "
        + "essentially low-pass filtered). A reasonable range for tuning it would probably be "
        + "[0.01, 10]; when set to 0.01 you should find that the magnitude of seasonality is "
        + "forced to be very small. This likely also makes sense on a log scale, since it is "
        + "effectively an L2 penalty like in ridge regression.",
    )

    daily_seasonality = st.sidebar.checkbox("Daily Seasonality", value=True)
    weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=True)
    yearly_seasonality = st.sidebar.selectbox(
        "Yearly Seasonality",
        (0, 1, 2),
        format_func=lambda x: {
            0: "No Yearly Seasonality",
            1: "Yearly Seasonality",
            2: "Custom Yearly Seasonality",
        }[x],
    )

    perform_cross_validation = st.sidebar.selectbox(
        "Cross Validation", ("No", "1 days", "2 days")
    )

    model_robustness = st.sidebar.checkbox(
        "Model Robustness",
        help="R-squared (R2) is a statistical measure that represents the proportion of the "
        + "variance for a dependent variable that’s explained by an independent variable "
        + "or variables in a regression model."
        + "\n\r"
        + "Mean Absolute Error (MAE) measures the average magnitude of the errors in a set "
        + "of predictions, without considering their direction. It’s the average over the "
        + "test sample of the absolute differences between prediction and actual "
        + "observation where all individual differences have equal weight.",
    )

    hyperparameter_tuning = st.sidebar.checkbox("Hyperparameter Tuning")

    show_charts = st.sidebar.checkbox("Show Charts", value=True)

    if app_mode == "Run Site Forecast":
        single_site_forecast(
            selected_area,
            selected_category,
            days_of_usage,
            changepoint_prior_scale,
            changepoint_range,
            seasonality_mode,
            seasonality_prior_scale,
            daily_seasonality,
            weekly_seasonality,
            yearly_seasonality,
            perform_cross_validation,
            hyperparameter_tuning,
            model_robustness,
            category_specific_hyperparameters,
            site_specific_hyperparameters,
            show_charts,
        )

    if app_mode == "Run Full Forecast":
        full_forecast(
            selected_area,
            selected_category,
            days_of_usage,
            changepoint_prior_scale,
            changepoint_range,
            seasonality_mode,
            seasonality_prior_scale,
            daily_seasonality,
            weekly_seasonality,
            yearly_seasonality,
            perform_cross_validation,
            hyperparameter_tuning,
            model_robustness,
            category_specific_hyperparameters,
            site_specific_hyperparameters,
            show_charts,
        )

    if app_mode == "Customer Details":
        customer_details()

    if app_mode == "Forecasts":
        forecasts()


def single_site_forecast(
    selected_area,
    selected_category,
    days_of_usage,
    changepoint_prior_scale,
    changepoint_range,
    seasonality_mode,
    seasonality_prior_scale,
    daily_seasonality,
    weekly_seasonality,
    yearly_seasonality,
    perform_cross_validation,
    hyperparameter_tuning,
    model_robustness,
    category_specific_hyperparameters,
    site_specific_hyperparameters,
    show_charts,
):

    # Title and selections
    st.write("# Gas Usage Forecast - Single Site")

    #
    # Weather Stations lookup
    #

    if data_retrieval_method == "PANDAS_ACCESS":

        # Get data from access database
        weather_stations_df = mdb.read_table(MDB, "tblWeatherStation")

        projected_temp_df = mdb.read_table(MDB, "tblHeatingDDProjected")

        # Remove data not needed
        projected_temp_df = projected_temp_df.loc[
            pd.to_datetime(projected_temp_df["Date"]).dt.date >= forecast_for
        ]
        projected_temp_df["Date"] = pd.to_datetime(projected_temp_df["Date"])

        # Remove columns that we don't care about
        projected_temp_df = projected_temp_df[
            ["Date", "WeatherStationID", "AvgTemp"]
        ].dropna()

        # Make sure that we have projected temperatures for each area
        missing_temps = 0
        for index, row in areas_df.iterrows():

            if row["WeatherStationID"] not in projected_temp_df.values:

                st.warning(
                    "Missing projected temperature for "
                    + str(row["Description"])
                    + "!  Has the weather forecast been run?"
                )
                missing_temps = 1

    #
    # Site / Customer lookup
    #

    if data_retrieval_method == "PANDAS_ACCESS":

        sites_df = load_sites(MDB)

        # Remove sites that we aren't including in the forecast
        sites_df = sites_df[(sites_df.Include == "1")]
        sites_df.sort_values(by=["Name"], inplace=True)

        # If we chose an area, only show sites in that area
        if selected_area > 0:
            sites_df = sites_df[(sites_df.AreaID == selected_area)]

        # If we chose a category, only show sites in that category
        if selected_category > 0:
            sites_df = sites_df[(sites_df.CategoryID == selected_category)]

        # Remove columns that we don't care about
        sites_df = sites_df[
            ["ID", "Name", "AreaID", "CategoryID", "LDCActNum"]
        ].dropna()

        sites_values = []
        for index, row in sites_df.iterrows():
            sites_values.append(
                row["Name"]
                + " | "
                + str(row["LDCActNum"])
                + " ("
                + str(row["ID"])
                + ")"
            )

        sites_values = sites_df["Name"].tolist()
        sites_values.insert(0, "Select a site")
        sites_options = sites_df["ID"].tolist()
        sites_options.insert(0, 0)

        sites_dic = dict(zip(sites_options, sites_values))

    selected_site = st.selectbox(
        "Select a site", sites_options, format_func=lambda x: sites_dic[x]
    )

    if selected_site == 0:
        st.write("Please select a site above")
        return

    col1, col2 = st.beta_columns(2)

    with col1:

        category_id = sites_df.loc[sites_df["ID"] == selected_site]["CategoryID"].item()
        st.write(categories_df.loc[categories_df["CategoryID"] == category_id])
        st.write(selected_site)

        # Don't include weather regression for certain categories
        #
        # CategoryID    Description              Include Weather Regression?
        #
        #  1            Heating                  YES
        #  2            Asphalt Plant            NO
        #  3            Process                  NO
        #  5            Process (Heat Sensitive) YES

    with col2:
        area_id = sites_df.loc[sites_df["ID"] == selected_site]["AreaID"].item()
        st.write(areas_df.loc[areas_df["AreaID"] == area_id])

    default_weather_regression = True
    if category_id == 2 or category_id == 3:
        default_weather_regression = False
    apply_weather_regression = st.checkbox(
        "Apply weather regression", value=default_weather_regression
    )

    ## Get usage data

    if data_retrieval_method == "PYODBC":

        # query = "SELECT Date, Use FROM tblUse WHERE SiteID = " + selected_site + " ORDER BY Date"
        # df_gas = pd.read_sql(query, conn)
        # conn.close()
        print(data_retrieval_method)

    elif data_retrieval_method == "PANDAS_ACCESS":

        with st.spinner("Loading usage..."):

            df_gas = load_usage(MDB)

            # Remove data for sites not selected
            df_gas = df_gas[(df_gas.SiteID == selected_site)]

            # Remove columns that we don't care about
            df_gas = df_gas[["Date", "Use"]].dropna()

    # Renaming Columns for simplicity
    df_gas.rename(columns={"Date": "ds", "Use": "y"}, inplace=True)

    df_weather = []
    WeatherStationID = ""

    if apply_weather_regression or hyperparameter_tuning:

        with st.spinner("Retrieving weather..."):

            # Weather regressor
            # Dataset - tblHeatingDD ( WeatherStationID: 1 )

            sites_row = sites_df.loc[sites_df["ID"] == selected_site]
            areas_row = areas_df.loc[areas_df["AreaID"] == sites_row["AreaID"].item()]

            with col2:
                WeatherStationID = areas_row["WeatherStationID"].item()
                st.write(
                    (
                        weather_stations_df.loc[
                            weather_stations_df["WeatherStationID"] == WeatherStationID
                        ]
                    )["Description"]
                )

            # WeatherStationID  City        Pools
            # 1                 Binghamton  TCO
            # 2                 Buffalo     TGP
            # 3                 Syracuse    NIMO/NGRID
            # 4                 Plattsburgh NCPL
            # 5                 Rochester   DTI/RG&E

            if data_retrieval_method == "PANDAS_ACCESS":

                df_weather = load_weather(MDB)

                # Remove data for areas not selected
                df_weather = df_weather[
                    (df_weather.WeatherStationID == WeatherStationID)
                ]

                # Remove columns that we don't care about
                df_weather = df_weather[["Date", "AvgTemp"]].dropna()

                df_weather.rename(columns={"Date": "ds", "AvgTemp": "y"}, inplace=True)

            df_weather["date_index"] = df_weather["ds"]
            df_weather["date_index"] = pd.to_datetime(df_weather["ds"])
            df_weather = df_weather.set_index("date_index")

            df_gas["temp"] = pd.to_datetime(df_gas["ds"]).map(df_weather["y"])

    # Asphalt plant
    if category_specific_hyperparameters is True and category_id == 2:
        changepoint_prior_scale = 0.5
        days_of_usage = 15
        st.write("CAT SPEC HPARAM")

    # Process
    if category_specific_hyperparameters is True and category_id == 3:
        changepoint_prior_scale = 0.5
        days_of_usage = 30

    if site_specific_hyperparameters is True:

        # Connect to SQLite Database to store forecast information
        conn = sqlite3.connect(forecasting_db_location, timeout=10, uri=True)

        sss = site_specific_settings(
            conn,
            selected_site,
            days_of_usage,
            changepoint_prior_scale,
            changepoint_range,
            seasonality_mode,
            seasonality_prior_scale,
            daily_seasonality,
            weekly_seasonality,
            yearly_seasonality,
            apply_weather_regression,
        )

        (
            days_of_usage,
            changepoint_prior_scale,
            changepoint_range,
            seasonality_mode,
            seasonality_prior_scale,
            daily_seasonality,
            weekly_seasonality,
            yearly_seasonality,
            apply_weather_regression,
        ) = sss.retrieve()

    # convert the 'Date' column to datetime format, and sort
    df_gas["ds"] = pd.to_datetime(df_gas["ds"])
    df_gas.sort_values(by=["ds"], inplace=True)

    # Remove entries older than X days
    removeolderthan = pd.to_datetime(today_date) - datetime.timedelta(
        int(days_of_usage)
    )
    df_gas = df_gas.loc[df_gas["ds"] >= removeolderthan]

    df_gas = df_gas.loc[
        df_gas["ds"] <= pd.to_datetime(pd.to_datetime(today_date).date())
    ]

    with st.beta_expander("Show parameters used"):
        st.write("Days of usage " + str(days_of_usage))
        st.write("CPS " + str(changepoint_prior_scale))
        st.write("Changepoint Range " + str(changepoint_range))
        st.write("Seasonality Mode " + str(seasonality_mode))
        st.write("Seasonality Prior Scale " + str(seasonality_prior_scale))
        st.write("Daily Seasonality " + str(daily_seasonality))
        st.write("Weekly Seasonality " + str(weekly_seasonality))
        st.write("Yearly Seasonality " + str(yearly_seasonality))

    Prophet = predict_the_future(
        df_gas=df_gas,
        df_weather=df_weather,
        projected_temp_df=projected_temp_df,
        WeatherStationID=WeatherStationID,
        today_date=pd.to_datetime(today_date).date(),
        days_to_forecast=days_to_forecast,
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range=changepoint_range,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        apply_weather_regression=apply_weather_regression,
        streamlit=True,
    )

    Prophet.forecast(ph)
    m_gas = Prophet.m_gas
    forecast_gas = Prophet.forecast_gas

    if hyperparameter_tuning:

        # Use cross validation to evaluate all parameters
        # WITH weather regression applied
        for params in all_params:

            with suppress_stdout_stderr():

                m_gas_hp = ph.Prophet(**params)

                m_gas_hp.add_regressor("temp")  # Temperature regressor
                df_gas["temp"] = pd.to_datetime(df_gas["ds"]).map(df_weather["y"])

                # Drop columns with nan values, so we don't get an error
                df_gas.dropna(subset=["temp"], inplace=True)

                m_gas_hp.fit(df_gas)

            df_cv = cross_validation(m_gas_hp, horizon="3 days", parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p["rmse"].values[0])

        st.write(
            "Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit. Root mean square error is commonly used in climatology, forecasting, and regression analysis to verify experimental results."
        )

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results["rmse"] = rmses
        st.write("Tuning Results - Weather Regression")
        st.write(tuning_results)

        # Use cross validation to evaluate all parameters
        # WITHOUT weather regression applied
        for params in all_params:

            with suppress_stdout_stderr():

                m_gas_hp = ph.Prophet(**params)

                m_gas_hp.fit(df_gas)

            df_cv = cross_validation(m_gas_hp, horizon="3 days", parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p["rmse"].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results["rmse"] = rmses
        st.write("Tuning Results - NO Weather Regression")
        st.write(tuning_results)

    if perform_cross_validation != "No":

        ## Cross validation results
        # cv_results = cross_validation(m_gas, period = '30 days', horizon = perform_cross_validation)
        # cv_results = cross_validation(m_gas, period = '30 days', horizon = '7 days')
        cv_results = cross_validation(m_gas, horizon="7 days", parallel="processes")

        # Avoid divide by zero!
        cv_results = cv_results[cv_results.y != 0]

        st.write("Cross Validation Results")
        st.write(cv_results)

        # I struggled with this for a while as well. But here is how it works. The initial
        # model will be trained on the first 1,825 days of data. It will forecast the next
        # 60 days of data (because horizon is set to 60). The model will then train on the
        # initial period + the period (1,825 + 30 days in this case) and forecast the next
        # 60 days. It will continued like this, adding another 30 days to the training data
        # and then forecasting for the next 60 until there is no longer enough data to do
        # this.
        #
        # In summary, period is how much data to add to the training data set in every
        # iteration of cross-validation, and horizon is how far out it will forecast.

        ## Calculate the MAPE

        mape_baseline = mean_absolute_percentage_error(cv_results.y, cv_results.yhat)
        st.write("MAPE BASELINE: " + str(round(mape_baseline, 1)) + "%")

    if model_robustness:

        # https://medium.com/mlearning-ai/forecast-using-prophet-canadian-natural-gas-production-dataset-b1f9c57548d8
        # calculate MAE between expected and predicted values for next 60 mont
        smallest_entries = min(len(df_gas.index), len(forecast_gas.index))
        y_true = df_gas["y"][:(smallest_entries)].values
        y_pred = forecast_gas["yhat"][:(smallest_entries)].values

        # st.write(df_gas)
        # st.write(y_true)

        # st.write(forecast_gas)
        # st.write(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        st.write("MAE: %.3f" % mae)
        r = r2_score(y_true, y_pred)
        st.write("R-squared Score: %.3f" % r)

    # Why would we ever forecast negative gas??  I tried a lot of different ways (floor/cap)
    # but nothing other than the following suggestion seemed to work
    forecast_gas["yhat"] = np.where(forecast_gas["yhat"] < 0, 0, forecast_gas["yhat"])

    if show_charts:

        fig = plot_plotly(m_gas, forecast_gas)

        start_date = forecast_for - datetime.timedelta(30)
        end_date = forecast_for + datetime.timedelta(2)

        fig.update_xaxes(type="date", range=[start_date, end_date])

        fig.update_layout(title="Gas Usage", yaxis_title="Usage", xaxis_title="Date")
        st.plotly_chart(fig)

    # What does forecast_gas have for the next X days?
    forecast_gas_row = forecast_gas[forecast_gas["ds"] == forecast_for]
    value = forecast_gas_row.yhat.item()
    st.write(str(forecast_for) + " " + str(round(value, 1)))

    for i in range(days_to_forecast - 1):

        forecast_gas_row = forecast_gas[
            forecast_gas["ds"]
            == (pd.to_datetime(forecast_for) + datetime.timedelta(i + 1)).normalize()
        ]
        value = forecast_gas_row.yhat.item()
        st.write(
            str(
                (pd.to_datetime(forecast_for) + datetime.timedelta(i + 1))
                .normalize()
                .date()
            )
            + ": "
            + str(round(value, 1))
        )

    with st.beta_expander("See components"):

        ## Plot/Show components
        fig = m_gas.plot_components(forecast_gas)
        if apply_weather_regression:
            # https://facebook.github.io/prophet/docs/trend_changepoints.html
            a = add_changepoints_to_plot(fig.gca(), m_gas, forecast_gas)
        st.plotly_chart(fig)

        ## Go figure??
        import plotly.graph_objs as go

        fig = go.Figure()

        # Create and style traces
        fig.add_trace(go.Scatter(x=df_gas["ds"], y=df_gas["y"], name="Actual"))
        fig.add_trace(
            go.Scatter(x=forecast_gas["ds"], y=forecast_gas["yhat"], name="Predicted")
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_gas["ds"], y=forecast_gas["holidays"], name="Holidays"
            )
        )
        if apply_weather_regression:
            fig.add_trace(
                go.Scatter(x=df_gas["ds"], y=df_gas["temp"], name="Temperature")
            )
        st.plotly_chart(fig)

    if st.button("Run expensive computation"):
        a = 3
        b = 21
        res = expensive_computation(a, b)

        st.write("Result:", res)


def full_forecast(
    selected_area,
    selected_category,
    days_of_usage,
    changepoint_prior_scale,
    changepoint_range,
    seasonality_mode,
    seasonality_prior_scale,
    daily_seasonality,
    weekly_seasonality,
    yearly_seasonality,
    perform_cross_validation,
    hyperparameter_tuning,
    model_robustness,
    category_specific_hyperparameters,
    site_specific_hyperparameters,
    show_charts,
):

    progress_display = 1
    force_weather_regression_for_all = False

    # Start the stopwatch / counter
    timer_start = process_time()

    # Title and selections
    st.write("# Gas Usage Forecast - Full")

    #
    # Areas lookup
    #

    # Get data from access database
    areas_df = mdb.read_table(MDB, "tblArea")

    # Remove areas that we aren't including in the forecast
    areas_df = areas_df[(areas_df.IncludeOnReport == 1)]
    areas_df.sort_values(by=["Description"], inplace=True)

    # Remove columns that we don't care about
    areas_df = areas_df[["AreaID", "Description", "WeatherStationID"]].dropna()

    # Create our dictionary of dictionaries

    forecast_totals_by_area = {}

    for i in range(days_to_forecast):

        st.write(str((forecast_for + datetime.timedelta(i)).normalize().date()))
        forecast_totals_by_area[
            str((forecast_for + datetime.timedelta(i)).normalize().date())
        ] = {}

        for areas_index, areas_row in areas_df.iterrows():

            forecast_totals_by_area[
                str((forecast_for + datetime.timedelta(i)).normalize().date())
            ][areas_row["Description"]] = 0

    #
    # Pull all usage data
    #

    df_gas_master = mdb.read_table(MDB, "tblUse")

    #
    # Weather Stations lookup
    #

    # Get data from access database
    weather_stations_df = mdb.read_table(MDB, "tblWeatherStation")

    projected_temp_df = mdb.read_table(MDB, "tblHeatingDDProjected")

    # Remove data not needed
    projected_temp_df = projected_temp_df.loc[
        pd.to_datetime(projected_temp_df["Date"]).dt.date >= forecast_for
    ]
    projected_temp_df["Date"] = pd.to_datetime(projected_temp_df["Date"])

    # Make sure that we have projected temperatures for each area
    missing_temps = 0
    for index, row in areas_df.iterrows():

        if row["WeatherStationID"] not in projected_temp_df.values:

            st.warning(
                "Missing projected temperature for "
                + str(row["Description"])
                + "!  Has the weather forecast been run?"
            )
            missing_temps = 1

    if missing_temps:
        exit()

    # Remove columns that we don't care about
    projected_temp_df = projected_temp_df[
        ["Date", "WeatherStationID", "AvgTemp"]
    ].dropna()

    # If we have specified a date other than today, remove unnecessary weather data
    if today_date != "today":
        projected_temp_df = projected_temp_df.loc[
            projected_temp_df["Date"] <= pd.to_datetime(forecast_for)
        ]

    for areas_index, areas_row in areas_df.iterrows():

        # AreaID	Description
        # 1         NYSEG DTI
        # 2	        NYSEG TCO
        # 3         NYSEG TGP
        # 4	        National Grid East
        # 5	        National Grid West
        # 6         RG&E
        # 11        NYSEG NCPL
        # 12        NYSEG O&R
        # 13        NYSEG Olean

        if selected_area > 0 and areas_row["AreaID"] != selected_area:
            continue

        area_description = areas_row["Description"]

        # Get weather for weather regression, if needed
        # Dataset - tblHeatingDD ( WeatherStationID: 1 )

        # sites_row = sites_df.loc[sites_df['ID'] == sites_row['ID']]
        # areas_row = areas_df.loc[areas_df['AreaID'] == areas_row['AreaID'].item()]

        WeatherStationID = areas_row["WeatherStationID"]

        # WeatherStationID  City        Pools
        # 1                 Binghamton  TCO
        # 2                 Buffalo     TGP
        # 3                 Syracuse    NIMO/NGRID
        # 4                 Plattsburgh NCPL
        # 5                 Rochester   DTI/RG&E

        # Get data from access database
        df_weather_master = mdb.read_table(MDB, "tblHeatingDD")

        # Remove data for dates not needed
        if days_of_usage != "ALL":
            df_weather_master = df_weather_master.loc[
                pd.to_datetime(df_weather_master["Date"]).dt.date
                >= (forecast_for - datetime.timedelta(days_of_usage))
            ]

        #
        # Site / Customer lookup
        #

        # Get data from access database
        sites_df = mdb.read_table(MDB, "Sites")

        # Remove sites that we aren't including in the forecast
        sites_df = sites_df[(sites_df.Include == "1")]
        sites_df.sort_values(by=["Name"], inplace=True)

        # Filter out sites in area that we are processing
        sites_df = sites_df[(sites_df.AreaID == areas_row["AreaID"])]

        # Remove columns that we don't care about
        sites_df = sites_df[["ID", "Name", "AreaID", "CategoryID"]].dropna()

        # Remove weather data for areas not selected
        df_weather = df_weather_master[
            (df_weather_master.WeatherStationID == WeatherStationID)
        ]

        # Remove columns that we don't care about
        df_weather = df_weather[["Date", "AvgTemp"]].dropna()

        df_weather.rename(columns={"Date": "ds", "AvgTemp": "y"}, inplace=True)

        df_weather["date_index"] = df_weather["ds"]
        df_weather["date_index"] = pd.to_datetime(df_weather["ds"])

        # Simulate another date's available data if not today
        if today_date != "today":
            df_weather = df_weather.loc[
                df_weather["date_index"]
                <= pd.to_datetime(pd.to_datetime(today_date).date())
            ]

        df_weather = df_weather.set_index("date_index")

        ##
        ## Print information
        ##

        if progress_display:
            st.write("=" * 50)
            st.write(areas_row["AreaID"], area_description)
            st.write(str(len(sites_df.index)) + " sites to process in this area")
            st.write(
                projected_temp_df.loc[
                    projected_temp_df["WeatherStationID"] == WeatherStationID
                ]
            )
            st.write("=" * 50)

        st.write("Weather regression data:")
        st.dataframe(df_weather)

        for line_number, (sites_index, sites_row) in enumerate(sites_df.iterrows()):

            if selected_category > 0 and sites_row["CategoryID"] != selected_category:
                continue

            if progress_display:
                st.write(
                    "{} {}  [ {}/{} {:.2f}% complete ]".format(
                        sites_row["ID"],
                        sites_row["Name"],
                        (line_number + 1),
                        len(sites_df.index),
                        100 * (line_number + 1) / len(sites_df),
                    )
                )

            apply_weather_regression = False
            if (
                sites_row["CategoryID"] == 1
                or sites_row["CategoryID"] == 4
                or force_weather_regression_for_all
            ):
                apply_weather_regression = True

            if progress_display:
                apply_weather_regression_color = "red"
                if apply_weather_regression:
                    apply_weather_regression_color = "green"
                st.markdown(
                    "&nbsp;&nbsp;&nbsp;&nbsp; Applying weather regression? <font color='"
                    + apply_weather_regression_color
                    + "'>"
                    + str(apply_weather_regression)
                    + "</font><br>",
                    unsafe_allow_html=True,
                )

            ## Get usage data

            # Remove data for sites not selected
            df_gas = df_gas_master[(df_gas_master.SiteID == sites_row["ID"])]

            # Remove columns that we don't care about
            df_gas = df_gas[["Date", "Use"]].dropna()

            # Renaming Columns for simplicity
            df_gas.rename(columns={"Date": "ds", "Use": "y"}, inplace=True)

            # convert the 'Date' column to datetime format, and sort
            df_gas["ds"] = pd.to_datetime(df_gas["ds"])
            df_gas.sort_values(by=["ds"], inplace=True)

            # Just in case we changed the forecast date, remove usage after to avoid problems
            if today_date == "today":
                df_gas = df_gas.loc[
                    df_gas["ds"] <= pd.to_datetime(pd.to_datetime(today_date).date())
                ]
            else:
                df_gas = df_gas.loc[
                    df_gas["ds"] < pd.to_datetime(pd.to_datetime(today_date).date())
                ]

            if apply_weather_regression:

                df_gas["temp"] = pd.to_datetime(df_gas["ds"]).map(df_weather["y"])

            temp_days_of_usage = days_of_usage
            temp_changepoint_range = 0.80
            temp_changepoint_prior_scale = changepoint_prior_scale
            temp_seasonality_mode = "additive"
            temp_seasonality_prior_scale = 10
            temp_daily_seasonality = daily_seasonality
            temp_weekly_seasonality = weekly_seasonality
            temp_yearly_seasonality = yearly_seasonality

            # Asphalt plant
            if (
                category_specific_hyperparameters is True
                and sites_row["CategoryID"] == 2
            ):
                temp_changepoint_prior_scale = 0.5
                temp_days_of_usage = 15

            # Process
            if (
                category_specific_hyperparameters is True
                and sites_row["CategoryID"] == 3
            ):
                temp_changepoint_prior_scale = 0.5
                temp_days_of_usage = 30

            if site_specific_hyperparameters is True:

                # Connect to SQLite Database to store forecast information
                conn = sqlite3.connect(forecasting_db_location, timeout=10, uri=True)

                site_params_df = pd.read_sql_query(
                    "SELECT setting, value "
                    + "FROM Site_Forecast_Settings "
                    + "WHERE site_id = "
                    + str(sites_row["ID"]),
                    conn,
                )

                sss = site_specific_settings(
                    conn,
                    sites_row["ID"],
                    temp_days_of_usage,
                    temp_changepoint_prior_scale,
                    temp_changepoint_range,
                    temp_seasonality_mode,
                    temp_seasonality_prior_scale,
                    temp_daily_seasonality,
                    temp_weekly_seasonality,
                    temp_yearly_seasonality,
                    apply_weather_regression,
                )

                (
                    temp_days_of_usage,
                    temp_changepoint_prior_scale,
                    temp_changepoint_range,
                    temp_seasonality_mode,
                    temp_seasonality_prior_scale,
                    temp_daily_seasonality,
                    temp_weekly_seasonality,
                    temp_yearly_seasonality,
                    apply_weather_regression,
                ) = sss.retrieve()

                st.dataframe(site_params_df)

            # Remove entries older than X days
            if days_of_usage != "ALL":
                removeolderthan = forecast_for - datetime.timedelta(
                    int(temp_days_of_usage)
                )
                df_gas = df_gas.loc[df_gas["ds"] >= pd.to_datetime(removeolderthan)]

            Prophet = predict_the_future(
                df_gas=df_gas,
                df_weather=df_weather,
                projected_temp_df=projected_temp_df,
                WeatherStationID=WeatherStationID,
                today_date=pd.to_datetime(today_date).date(),
                days_to_forecast=days_to_forecast,
                changepoint_prior_scale=temp_changepoint_prior_scale,
                changepoint_range=temp_changepoint_range,
                seasonality_mode=temp_seasonality_mode,
                seasonality_prior_scale=temp_seasonality_prior_scale,
                daily_seasonality=temp_daily_seasonality,
                weekly_seasonality=temp_weekly_seasonality,
                yearly_seasonality=temp_yearly_seasonality,
                apply_weather_regression=apply_weather_regression,
                streamlit=True,
            )

            Prophet.forecast(ph)

            m_gas = Prophet.m_gas
            forecast_gas = Prophet.forecast_gas

            if model_robustness:

                # https://medium.com/mlearning-ai/forecast-using-prophet-canadian-natural-gas-production-dataset-b1f9c57548d8
                # calculate MAE between expected and predicted values for next 60 mont
                smallest_entries = min(len(df_gas.index), len(forecast_gas.index))
                y_true = df_gas["y"][:(smallest_entries)].values
                y_pred = forecast_gas["yhat"][:(smallest_entries)].values

                # st.write(df_gas)
                # st.write(y_true)

                # st.write(forecast_gas)
                # st.write(y_pred)

                mae = mean_absolute_error(y_true, y_pred)
                st.write("MAE: %.3f" % mae)
                r = r2_score(y_true, y_pred)
                st.write("R-squared Score: %.3f" % r)

            ## Cross validation results
            # cv_results = cross_validation(m_gas, initial = '30 days', period = '30 days', horizon = '30 days')
            ## Calculate the MAPE
            # mape_baseline = mean_absolute_percentage_error(cv_results.y, cv_results.yhat)
            # st.write("MAPE BASELINE: " + str(mape_baseline))

            # Why would we ever forecast negative gas??  I tried a lot of different ways (floor/cap)
            # but nothing other than the following suggestion seemed to work
            forecast_gas["yhat"] = np.where(
                forecast_gas["yhat"] < 0, 0, forecast_gas["yhat"]
            )

            # What does forecast_gas have for the next X days?

            forecast_gas_row = forecast_gas[forecast_gas["ds"] == str(forecast_for)]
            value = forecast_gas_row.yhat.item()
            if value < 0:
                value = 0
            if progress_display:
                st.write(
                    "&nbsp;&nbsp;&nbsp;&nbsp; "
                    + str(forecast_for.normalize().date())
                    + ": "
                    + str(round(value, 1))
                )
            forecast_totals_by_area[str(forecast_for.normalize().date())][
                area_description
            ] += value

            for i in range(days_to_forecast - 1):

                forecast_gas_row = forecast_gas[
                    forecast_gas["ds"]
                    == (
                        pd.to_datetime(forecast_for) + datetime.timedelta(i + 1)
                    ).normalize()
                ]
                value = forecast_gas_row.yhat.item()

                # Prophet sometimes forecasts less than 0.  Doesn't look easy to correct this,
                # so for now I am correcting it here!
                if value < 0:
                    value = 0
                forecast_date = (
                    (
                        pd.to_datetime(forecast_for.normalize().date())
                        + datetime.timedelta(i + 1)
                    )
                    .normalize()
                    .date()
                )

                if progress_display:
                    st.write(
                        "&nbsp;&nbsp;&nbsp;&nbsp; "
                        + str(forecast_date)
                        + ": "
                        + str(round(value, 1))
                    )

                forecast_totals_by_area[str(forecast_date)][area_description] += value

            # Why would we ever forecast negative gas??  I tried a lot of different ways (floor/cap)
            # but nothing other than the following suggestion seemed to work
            forecast_gas["yhat"] = np.where(
                forecast_gas["yhat"] < 0, 0, forecast_gas["yhat"]
            )

            if show_charts:
                fig = plot_plotly(m_gas, forecast_gas)

                start_date = forecast_for - datetime.timedelta(30)
                end_date = forecast_for + datetime.timedelta(2)

                fig.update_xaxes(type="date", range=[start_date, end_date])

                fig.update_layout(
                    title="Gas Usage", yaxis_title="Usage", xaxis_title="Date"
                )

                st.plotly_chart(fig)

            st.write("-" * 50)

    st.write("=" * 50)
    st.write("Days of usage: " + str(days_of_usage))
    st.write("Changepoint Prior Scale (CPS): " + str(changepoint_prior_scale))
    st.write("=" * 50)

    for i in range(days_to_forecast):

        print_date = str((forecast_for + datetime.timedelta(i)).normalize().date())

        st.write(print_date)

        st.write(round(forecast_totals_by_area[print_date]["National Grid West"]))
        st.write(round(forecast_totals_by_area[print_date]["NYSEG DTI"]))
        st.write(round(forecast_totals_by_area[print_date]["NYSEG NCPL"]))
        st.write(round(forecast_totals_by_area[print_date]["NYSEG O&R"]))
        st.write(round(forecast_totals_by_area[print_date]["NYSEG Olean"]))
        st.write(round(forecast_totals_by_area[print_date]["NYSEG TCO"]))
        st.write(round(forecast_totals_by_area[print_date]["NYSEG TGP"]))
        st.write(round(forecast_totals_by_area[print_date]["RG&E"]))

    # Stop the stopwatch / counter
    timer_stop = process_time()

    st.write("Elapsed time:", (timer_stop - timer_start) / 60)


def customer_details():

    st.write("# Customer Details")

    # Get data from access database
    weather_stations_df = mdb.read_table(MDB, "tblWeatherStation")

    sites_df = load_sites(MDB)

    # Get data from access database
    categories_df = mdb.read_table(MDB, "tblUseCategory")

    # Remove sites that we aren't including in the forecast
    sites_df = sites_df[(sites_df.Include == "1")]
    sites_df.sort_values(by=["Name"], inplace=True)

    st.write(str(len(sites_df.index)) + " Total Customers")

    areas_df = mdb.read_table(MDB, "tblArea")

    # Remove areas that we aren't including in the forecast
    areas_df = areas_df[(areas_df.IncludeOnReport == 1)]
    areas_df.sort_values(by=["Description"], inplace=True)

    for index, row in areas_df.iterrows():

        st.header(
            "( "
            + str(row["AreaID"])
            + " ) "
            + areas_df.loc[areas_df["AreaID"] == row["AreaID"]].Description.item()
            + " : "
            + weather_stations_df.loc[
                weather_stations_df["WeatherStationID"]
                == areas_df.loc[
                    areas_df["AreaID"] == row["AreaID"]
                ].WeatherStationID.item()
            ].Description.item()
        )

        filtered_sites_df = sites_df[(sites_df.AreaID == row["AreaID"])]

        total_customer_count = filtered_sites_df.ID.count()

        heating_site_count = filtered_sites_df[
            (filtered_sites_df.CategoryID == 1)
        ].ID.count()

        asphalt_site_count = filtered_sites_df[
            (filtered_sites_df.CategoryID == 2)
        ].ID.count()

        process_site_count = filtered_sites_df[
            (filtered_sites_df.CategoryID == 3)
        ].ID.count()

        proc_heat_site_count = filtered_sites_df[
            (filtered_sites_df.CategoryID == 5)
        ].ID.count()

        customer_details_df = pd.DataFrame(
            [
                [
                    "Heating",
                    heating_site_count,
                    "{:.0%}".format(heating_site_count / total_customer_count),
                ],
                [
                    "Asphalt",
                    asphalt_site_count,
                    "{:.0%}".format(asphalt_site_count / total_customer_count),
                ],
                [
                    "Process",
                    process_site_count,
                    "{:.0%}".format(process_site_count / total_customer_count),
                ],
                [
                    "Process (Heat Sensitive)",
                    proc_heat_site_count,
                    "{:.0%}".format(proc_heat_site_count / total_customer_count),
                ],
                [
                    "TOTAL",
                    total_customer_count,
                    "100%",
                ],
            ],
            columns=["Category", "Count", "Percent"],
        )

        st.write(customer_details_df)
        repl = categories_df.set_index("CategoryID")["Description"]
        filtered_sites_df.replace(repl, inplace=True)

        with st.beta_expander(
            str(areas_df.loc[areas_df["AreaID"] == row["AreaID"]].Description.item())
            + " Customer Details"
        ):
            st.dataframe(
                filtered_sites_df[["Name", "LDCActNum", "CategoryID"]], height=10000
            )


def forecasts():

    # Connect to SQLite Database to store forecast information
    conn = sqlite3.connect(forecasting_db_location, timeout=10, uri=True)

    # Title and selections
    st.write("# Forecasts")

    sites_df = load_sites(MDB)
    gas_df = mdb.read_table(MDB, "tblUse")

    forecast_dates_df = pd.read_sql_query(
        "SELECT DISTINCT(Date(ForecastDate)) FROM Forecast_Summary "
        "ORDER BY ForecastDate DESC",
        conn,
    )

    selected_forecast_date = str(
        st.date_input(
            "Select a Forecast Date",
            value=pd.to_datetime(forecast_dates_df.iloc[0].item()),
            min_value=pd.to_datetime(forecast_dates_df.iloc[-1].item()),
            max_value=pd.to_datetime(forecast_dates_df.iloc[0].item()),
            help="The forecast date",
        )
    )

    gas_df = gas_df.loc[
        pd.to_datetime(gas_df["Date"]).dt.date == pd.to_datetime(selected_forecast_date)
    ]

    # Get total usage by area...
    # Map area id to gas_df, using site_id - then group by area_id

    total_usage_by_area_df = (
        gas_df.merge(sites_df, left_on="SiteID", right_on="ID")
        .reindex(columns=["AreaID", "Use"])
        .groupby(["AreaID"])
        .sum()
    )

    forecast_ids_df = pd.read_sql_query(
        "SELECT DISTINCT(fs.forecast_id), ft.ID "
        + "FROM Forecast_Summary AS fs "
        + "LEFT JOIN Forecasts f ON f.forecast_id = fs.forecast_id "
        + "LEFT JOIN Forecast_Types ft ON ft.ID = f.forecast_type_id "
        + "WHERE Date(fs.ForecastDate) = '"
        + selected_forecast_date
        + "' AND ft.Hidden != 1 ORDER BY CAST(ft.ID AS integer)",
        conn,
    )
    forecast_ids_df.sort_values(by=["ID"])

    list_of_forecast_ids = forecast_ids_df["forecast_id"].tolist()

    # Output forecast types

    forecast_types_df = pd.read_sql_query(
        "SELECT ID, Description " + "FROM Forecast_Types" + " ORDER BY ID",
        conn,
    )

    with st.beta_expander("See Forecast Types"):
        st.dataframe(forecast_types_df)

    # Output forecast summary

    forecast_summary_df = pd.read_sql_query(
        "SELECT fs.forecast_id AS forecast_id, ft.ID AS TypeID, AreaID, fs.Description AS Area, ForecastTotal "
        + "FROM Forecast_Summary AS fs "
        + "LEFT JOIN Forecasts AS f ON f.forecast_id = fs.forecast_id "
        + "LEFT JOIN Forecast_Types AS ft ON ft.ID = f.forecast_type_id "
        + "WHERE fs.forecast_id IN ("
        + str(",".join(map(str, list_of_forecast_ids)))
        + ") AND ForecastDate = '"
        + selected_forecast_date
        + "' "
        + " ORDER BY fs.Description COLLATE NOCASE",
        conn,
    )

    # Pivot dataframe
    pivot_forecast_summary_df = forecast_summary_df.pivot_table(
        index=["Area", "AreaID"],
        columns=["TypeID", "forecast_id"],
        # columns="TypeID",
        values="ForecastTotal",
    )

    # Name our index column and add a new column for total usage
    pivot_forecast_summary_df.index.name = "Area"
    pivot_forecast_summary_df["TotalUsage"] = ""
    no_usage = False

    # Find and fill in TotalUsage for area/forecast
    for pfs_index, pfs_row in pivot_forecast_summary_df.iterrows():

        area_id = pfs_index[1]
        # forecast_type = pfs_row.index[0][0]
        # forecast_id = pfs_row.index[0][1]

        try:
            pivot_forecast_summary_df.at[pfs_index, "TotalUsage"] = int(
                total_usage_by_area_df.at[area_id, "Use"]
            )

        except:
            pivot_forecast_summary_df.at[pfs_index, "TotalUsage"] = None
            no_usage = True

    st.dataframe(pivot_forecast_summary_df)

    if no_usage is False:

        with st.beta_expander("See Forecast Accuracy"):

            # pivot_forecast_summary_df = pivot_forecast_summary_df.reset_index(
            #     level=1, drop=True
            # )

            # pivot_forecast_summary_df.reset_index(drop=True, inplace=True)

            # for i in range(len(pivot_forecast_summary_df.columns) - 1):
            for pfs_index, pfs_row in pivot_forecast_summary_df.iterrows():

                for col in range(len(pfs_row) - 1):

                    forecast_type = pfs_row.index[col][0]
                    forecast_id = pfs_row.index[col][1]

                    pivot_forecast_summary_df.at[
                        pfs_index, (forecast_type, forecast_id)
                    ] = abs(
                        pivot_forecast_summary_df.at[
                            pfs_index, (forecast_type, forecast_id)
                        ]
                        - int(pfs_row.at["TotalUsage"])
                    )

            # Add total row
            pivot_forecast_summary_df.loc["Total"] = pivot_forecast_summary_df.sum()

            st.dataframe(pivot_forecast_summary_df)

            # Remove total row, and transpose dataframe
            del pivot_forecast_summary_df["TotalUsage"]
            st.dataframe(pivot_forecast_summary_df.transpose())

    ##
    ## Output site-specific forecasts
    ##

    st.header("Site-Specific Forecasts:")

    for areas_index, areas_row in areas_df.iterrows():

        with st.beta_expander(areas_row["Description"]):

            for forecast_ids_index, forecast_ids_row in forecast_ids_df.iterrows():

                st.write(
                    str(forecast_ids_row[1])
                    + " - forecast_id: "
                    + str(forecast_ids_row[0])
                )

                site_usage_df = pd.read_sql_query(
                    "SELECT site_id, ForecastAmount FROM Forecast_Detail "
                    + "WHERE forecast_id = "
                    + str(forecast_ids_row[0])
                    + " AND AreaID = "
                    + str(areas_row["AreaID"])
                    + " AND ForecastDate = '"
                    + selected_forecast_date
                    + "'",
                    conn,
                )

                # Copy site_id into new column, site_name
                site_usage_df["site_name"] = site_usage_df["site_id"]
                # Set the dataframe index to the site_name column
                site_usage_df.set_index("site_name", inplace=True)

                # Map and update the usage from the gas_df dataframe
                site_usage_df["UsageAmount"] = 0
                site_usage_df["UsageAmount"].update(gas_df.set_index("SiteID")["Use"])
                # Reset the index
                site_usage_df = site_usage_df.reset_index()

                # Replace site_name column with actual name from sites_df dataframe
                repl = sites_df.set_index("ID")["Name"]
                site_usage_df["site_name"].replace(repl, inplace=True)

                # Create new column showing the forecast and usage difference
                site_usage_df["Difference"] = abs(
                    site_usage_df["ForecastAmount"] - site_usage_df["UsageAmount"]
                )

                # if (
                #     site_usage_df["Difference"].all()
                #     == site_usage_df["ForecastAmount"].all()
                # ):
                #     site_usage_df["Difference"] = 0

                # Round some values
                # site_usage_df["Difference"].round(decimals=2)
                pd.set_option("display.precision", 1)
                st.dataframe(
                    site_usage_df.style.format(
                        {"ForecastAmount": "{:.1f}", "UsageAmount": "{:.1f}"}
                    )
                )

                # st.dataframe(site_usage_df)


def update_in_alist(alist, key, value):
    return [(k, v) if (k != key) else (key, value) for (k, v) in alist]


def update_in_alist_inplace(alist, key, value):
    alist[:] = update_in_alist(alist, key, value)


def mean_absolute_percentage_error(y_true, y_pred):
    # Take in true and predicted values and calculate the MAPE score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print(y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


@st.cache(suppress_st_warning=True)
def expensive_computation(a, b):
    st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")

    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)  # 👈 This makes the function take 2s to run
        my_bar.progress(percent_complete + 1)

    return a * b


@st.cache(ttl=3600)
def load_sites(MDB):

    # Get data from access database
    sites_df = mdb.read_table(MDB, "Sites")

    # Remove sites that we aren't including in the forecast
    sites_df = sites_df[(sites_df.Include == "1")]
    sites_df.sort_values(by=["Name"], inplace=True)

    return sites_df


@st.cache(ttl=3600)
def load_usage(MDB):

    # Get data from access database
    df_gas = mdb.read_table(MDB, "tblUse")

    return df_gas


@st.cache(ttl=3600)
def load_weather(MDB):

    # Get data from access database
    df_weather = mdb.read_table(MDB, "tblHeatingDD")

    return df_weather


import os


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


if __name__ == "__main__":

    main()