import pandas as pd
import pandas_access as mdb

import numpy as np
import fbprophet as ph
import distutils

from fbprophet.diagnostics import cross_validation

import datetime
import sys
from time import process_time

import pytz

eastern = pytz.timezone("US/Eastern")

import pathlib
from apps.utils import predict_the_future, site_specific_settings

import sqlite3
from sqlite3 import Error

# Configuration Stuff

today_date = "today"
# today_date = "6/16/2021"

forecast_for = pd.to_datetime(today_date).date() + datetime.timedelta(1)

days_to_forecast = 1
if pd.to_datetime(today_date).weekday() == 4:
    days_to_forecast = 3  # Weekend!

progress_display = 1

days_of_usage = 30
changepoint_prior_scale = 0.05  # Default is 0.05

force_weather_regression_for_all = False

MDB = "/mnt/arminius/DailyBalancing_be.mdb"
DRV = "{Microsoft Access Driver (*.mdb)}"
USR = ""
PWD = ""

# Process command line arguments, if sent
if len(sys.argv) > 1:
    (
        script_name,
        forecast_type_id,
        progress_display,
        insert_into_db,
        growth,
        days_of_usage,
        changepoint_prior_scale,
        site_specific_hyperparameters,
        daily_seasonality,
        weekly_seasonality,
        yearly_seasonality,
    ) = sys.argv
    progress_display = int(progress_display)
    insert_into_db = int(insert_into_db)
    yearly_seasonality = int(yearly_seasonality)
    if days_of_usage != "ALL":
        days_of_usage = int(days_of_usage)
    changepoint_prior_scale = float(changepoint_prior_scale)

# Start the stopwatch / counter
timer_start = process_time()


def mean_absolute_percentage_error(y_true, y_pred):
    # Take in true and predicted values and calculate the MAPE score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def create_connection(db_file):
    """create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file, timeout=10)
    except Error as e:
        print(e)

    return conn


# Connect to SQLite Database to store forecast information
conn = create_connection("/mnt/arminius/Backend/Forecast_Data.db")


def create_forecast(conn, forecast_type_id):
    """
    Create a new forecast into the Forecasts table
    :param conn:
    :param forecast_type_id:
    :return: forecast id
    """
    sql = """ INSERT INTO Forecasts (forecast_type_id)
              VALUES (?)"""
    cur = conn.cursor()
    cur.execute(sql, [forecast_type_id])
    conn.commit()
    return cur.lastrowid


def finish_forecast(conn, forecast_id):

    sql = """ UPDATE Forecasts SET FinishDate = datetime('now')
                WHERE forecast_id = ? """
    cur = conn.cursor()
    cur.execute(sql, [forecast_id])
    conn.commit()
    return cur.lastrowid


def save_site_forecast(conn, forecast):
    """
    Save a new forecast into the Forecast_Detail table
    :param conn:
    :param project:
    :return: forecast id
    """
    sql = """ INSERT INTO Forecast_Detail (forecast_id, ForecastDate, AreaID, site_id, ForecastAmount)
              VALUES(?, ?, ?, ?, ?) """
    cur = conn.cursor()
    cur.execute(sql, forecast)
    conn.commit()
    return cur.lastrowid


def save_forecast_summary(conn, forecast):
    """
    Save a new forecast summary into the Forecast_Summary table
    :param conn:
    :param project:
    :return: forecast id
    """
    sql = """ INSERT INTO Forecast_Summary (forecast_id, ForecastDate, AreaID, Description, ForecastTotal)
              VALUES(?, ?, ?, ?, ?) """
    cur = conn.cursor()
    cur.execute(sql, forecast)
    conn.commit()
    return cur.lastrowid


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

    forecast_totals_by_area[str((forecast_for + datetime.timedelta(i)))] = {}

    for areas_index, areas_row in areas_df.iterrows():

        forecast_totals_by_area[str((forecast_for + datetime.timedelta(i)))][
            areas_row["Description"]
        ] = 0

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
## 6/1/2021 - it appears that on the 1st of the month, the weather importer
## program does NOT get historical temps for the 1st (which makes sense...)
## Trying to use projected temperature for the current day (rather than historic)
## This may cause issues on days that are not the 1st, or may make the forecast
## more/less accurate?

projected_temp_df = projected_temp_df.loc[
    pd.to_datetime(projected_temp_df["Date"]).dt.date >= pd.to_datetime(today_date)
]

projected_temp_df["Date"] = pd.to_datetime(projected_temp_df["Date"])

# Make sure that we have projected temperatures for each area
missing_temps = 0
for index, row in areas_df.iterrows():

    if row["WeatherStationID"] not in projected_temp_df.values:

        print(
            "Missing projected temperature for "
            + str(row["Description"])
            + "!  Has the weather forecast been run?"
        )
        missing_temps = 1

if missing_temps:
    exit()

# Remove columns that we don't care about
projected_temp_df = projected_temp_df[["Date", "WeatherStationID", "AvgTemp"]].dropna()

# Create a forecast_id to tie everything together
if insert_into_db:
    forecast_id = create_forecast(conn, forecast_type_id)

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

    # if areas_row["AreaID"] != 1:
    # continue

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
    df_weather = df_weather.set_index("date_index")

    ##
    ## Print information
    ##

    if progress_display:
        print("-" * 50)
        print(areas_row["AreaID"], area_description)
        print(str(len(sites_df.index)) + " sites to process in this area")
        print(
            projected_temp_df.loc[
                projected_temp_df["WeatherStationID"] == WeatherStationID
            ]
        )
        print("-" * 50)

    # print("Weather regression data:")
    # print(df_weather)

    for line_number, (sites_index, sites_row) in enumerate(sites_df.iterrows()):

        # if sites_row["ID"] != 410:
        #    continue

        if progress_display:
            print(
                "{} {}  [ {}/{} {:.2f}% complete ]".format(
                    sites_row["ID"],
                    sites_row["Name"],
                    (line_number + 1),
                    len(sites_df.index),
                    100 * (line_number + 1) / len(sites_df),
                )
            )

        ##
        ## Determine if we need to apply weather regression
        ## (Heating & Heating/Process categories)
        ##

        apply_weather_regression = False

        temp_days_of_usage = days_of_usage
        temp_growth = growth
        temp_changepoint_range = 0.80
        temp_changepoint_prior_scale = changepoint_prior_scale
        temp_seasonality_mode = "additive"
        temp_seasonality_prior_scale = 10
        temp_daily_seasonality = daily_seasonality
        temp_weekly_seasonality = weekly_seasonality
        temp_yearly_seasonality = yearly_seasonality

        if (
            sites_row["CategoryID"] == 1
            or sites_row["CategoryID"] == 4
            or force_weather_regression_for_all
        ):
            apply_weather_regression = True

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

        if site_specific_hyperparameters is True:

            sss = site_specific_settings(
                conn,
                sites_row["ID"],
                temp_days_of_usage,
                temp_growth,
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
                temp_growth,
                temp_changepoint_prior_scale,
                temp_changepoint_range,
                temp_seasonality_mode,
                temp_seasonality_prior_scale,
                temp_daily_seasonality,
                temp_weekly_seasonality,
                temp_yearly_seasonality,
                apply_weather_regression,
            ) = sss.retrieve()

        # Remove entries older than X days
        if days_of_usage != "ALL":
            removeolderthan = forecast_for - datetime.timedelta(int(temp_days_of_usage))
            df_gas = df_gas.loc[df_gas["ds"] >= pd.to_datetime(removeolderthan)]

        # Just in case we changed the forecast date, remove usage after to avoid problems
        df_gas = df_gas.loc[
            df_gas["ds"] <= pd.to_datetime(pd.to_datetime(today_date).date())
        ]

        if apply_weather_regression:

            df_gas["temp"] = pd.to_datetime(df_gas["ds"]).map(df_weather["y"])

        if progress_display:
            print("    Applying weather regression? " + str(apply_weather_regression))
            print("    Category " + str(sites_row["CategoryID"]))
            print("    Days of Usage " + str(temp_days_of_usage))
            print("    Growth " + str(temp_growth))
            print("    CPS " + str(temp_changepoint_prior_scale))
            print("    Changepoint Range " + str(temp_changepoint_range))
            print("    Seasonality Mode " + str(temp_seasonality_mode))
            print("    Seasonality Prior Scale " + str(temp_seasonality_prior_scale))
            print("    Daily Seasonality " + str(temp_daily_seasonality))
            print("    Weekly Seasonality " + str(temp_weekly_seasonality))
            print("    Yearly Seasonality " + str(temp_yearly_seasonality))

        Prophet = predict_the_future(
            df_gas=df_gas,
            df_weather=df_weather,
            projected_temp_df=projected_temp_df,
            WeatherStationID=WeatherStationID,
            today_date=pd.to_datetime(today_date).date(),
            days_to_forecast=days_to_forecast,
            growth=temp_growth,
            changepoint_prior_scale=temp_changepoint_prior_scale,
            changepoint_range=temp_changepoint_range,
            seasonality_mode=temp_seasonality_mode,
            seasonality_prior_scale=temp_seasonality_prior_scale,
            daily_seasonality=temp_daily_seasonality,
            weekly_seasonality=temp_weekly_seasonality,
            yearly_seasonality=temp_yearly_seasonality,
            apply_weather_regression=apply_weather_regression,
            streamlit=False,
        )

        Prophet.forecast(ph)
        m_gas = Prophet.m_gas
        forecast_gas = Prophet.forecast_gas

        # print(forecast_gas)

        ## Cross validation results
        # cv_results = cross_validation(m_gas, initial = '180 days', period = '7 days', horizon = '30 days')
        ## Calculate the MAPE
        # mape_baseline = mean_absolute_percentage_error(cv_results.y, cv_results.yhat)
        # print("MAPE BASELINE: " + str(mape_baseline))

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
            print("    " + str(forecast_for) + ": " + str(round(value, 1)))

        if insert_into_db:
            save_site_forecast(
                conn,
                (
                    forecast_id,
                    forecast_for,
                    areas_row["AreaID"],
                    sites_row["ID"],
                    round(value, 1),
                ),
            )

        forecast_totals_by_area[str(forecast_for)][area_description] += value

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
                (pd.to_datetime(forecast_for) + datetime.timedelta(i + 1))
                .normalize()
                .date()
            )

            if progress_display:
                print("    " + str(forecast_date) + ": " + str(round(value, 1)))

            if insert_into_db:
                save_site_forecast(
                    conn,
                    (
                        forecast_id,
                        forecast_date,
                        areas_row["AreaID"],
                        sites_row["ID"],
                        round(value, 1),
                    ),
                )

            forecast_totals_by_area[str(forecast_date)][area_description] += value

print("=" * 50)
print("Forecast Type: " + str(forecast_type_id))
print("Days of usage: " + str(days_of_usage))
print("Growth " + growth)
if site_specific_hyperparameters:
    print(
        "Changepoint Prior Scale (CPS): Site Specific [ "
        + str(changepoint_prior_scale)
        + " ]"
    )
else:
    print("Changepoint Prior Scale (CPS): " + str(changepoint_prior_scale))

print("Seasonality")
print("Daily: " + str(daily_seasonality))
print("Weekly: " + str(weekly_seasonality))
print("Yearly: " + str(yearly_seasonality))
print("=" * 50)

# import pprint
# pprint.pprint(forecast_totals_by_area)

formatted_list = list()

for i in range(days_to_forecast):

    print_date = str((forecast_for + datetime.timedelta(i)))

    print(print_date)

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

        area_description = areas_row["Description"]

        total = round(forecast_totals_by_area[print_date][area_description])
        if area_description == "NYSEG DTI":
            formatted_list.insert(1, total)
        if area_description == "NYSEG TCO":
            formatted_list.insert(5, total)
        if area_description == "NYSEG TGP":
            formatted_list.insert(6, total)
        if area_description == "National Grid West":
            formatted_list.insert(0, total)
        if area_description == "RG&E":
            formatted_list.insert(7, total)
        if area_description == "NYSEG NCPL":
            formatted_list.insert(2, total)
        if area_description == "NYSEG O&R":
            formatted_list.insert(3, total)
        if area_description == "NYSEG Olean":
            formatted_list.insert(4, total)

        if insert_into_db:
            forecast_summary_id = save_forecast_summary(
                conn,
                (
                    forecast_id,
                    print_date,
                    areas_row["AreaID"],
                    area_description,
                    round(forecast_totals_by_area[print_date][area_description]),
                ),
            )

    print(*formatted_list, sep="\n")
    formatted_list.clear()


# Stop the stopwatch / counter
timer_stop = process_time()

print("Elapsed time:", timer_stop - timer_start)
if insert_into_db:
    finish_forecast(conn, forecast_id)

########################################################
