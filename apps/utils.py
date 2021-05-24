import pandas as pd
import datetime
import holidays
import os
import streamlit as st  # pylint: disable=import-error

# @st.cache
class predict_the_future(object):
    def __init__(
        self,
        df_gas,
        df_weather,
        projected_temp_df,
        WeatherStationID,
        today_date,
        days_to_forecast,
        changepoint_prior_scale,
        changepoint_range,
        seasonality_mode,
        seasonality_prior_scale,
        daily_seasonality,
        weekly_seasonality,
        yearly_seasonality,
        apply_weather_regression,
        streamlit,
    ):
        super(predict_the_future, self).__init__()

        self.df_gas = df_gas
        self.df_weather = df_weather
        self.projected_temp_df = projected_temp_df
        self.WeatherStationID = WeatherStationID

        self.today_date = today_date
        self.days_to_forecast = days_to_forecast

        self.changepoint_prior_scale = changepoint_prior_scale
        self.changepoint_range = changepoint_range

        self.seasonality_mode = seasonality_mode.lower()
        self.seasonality_prior_scale = seasonality_prior_scale
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality

        self.apply_weather_regression = apply_weather_regression

        self.streamlit = streamlit

        self.m_gas = ""
        self.forecast_gas = ""

    def is_weekend_or_weekday(self, ds):
        date = pd.to_datetime(ds)
        return date.weekday() > 4 or date.weekday() < 1

    def forecast(self, ph):

        holiday_list = [
            "Christmas Day",
            "Christmas Day (Observed)",
            "Thanksgiving",
            "Labor Day",
            "Independence Day",
            "Memorial Day",
            "New Year's Day",
            "New Year's Day (Observed)",
        ]

        holiday_list_dates = []

        for holiday in holidays.UnitedStates(
            years=[
                datetime.date.today().year - 1,
                datetime.date.today().year,
                datetime.date.today().year + 1,
            ]
        ).items():
            if holiday[1] in holiday_list:
                holiday_list_dates.append(holiday[0])

        # Python
        holiday_list = pd.DataFrame(
            {
                "holiday": "holiday_list",
                "ds": pd.to_datetime(holiday_list_dates),
                "lower_window": 0,
                "upper_window": 1,
            }
        )

        # Automatic FRI-SUN
        self.df_gas["weekend"] = self.df_gas["ds"].apply(self.is_weekend_or_weekday)
        self.df_gas["weekday"] = ~self.df_gas["ds"].apply(self.is_weekend_or_weekday)

        # Training time
        self.m_gas = (
            ph.Prophet(
                growth="linear",  # DEFAULT linear [ logistic ]
                holidays=holiday_list,
                # holidays_prior_scale=5,  # DEFAULT 10
                changepoint_prior_scale=float(
                    self.changepoint_prior_scale
                ),  # DEFAULT 0.05
                changepoint_range=float(self.changepoint_range),  # DEFAULT 0.8
                seasonality_prior_scale=self.seasonality_prior_scale,  # DEFAULT 10
                seasonality_mode=self.seasonality_mode,  # DEFAULT additive [ multiplicative ]
                # interval_width = 0.95,
                # mcmc_samples = 0,
                # Daily/Yearly seasonality?  (Fourier terms : yearly_seasonality=10 default)
            )
            # .add_seasonality(
            #     name="weekly_day", period=7, fourier_order=1, condition_name="weekend"
            # )
            # .add_seasonality(
            #     name="weekly_day", period=7, fourier_order=10, condition_name="weekday"
            # )
            # .add_seasonality(
            #     name="yearly",
            #     period=365,
            #     fourier_order=3,
            #     prior_scale=10,
            #     mode="additive",
            # )
            # .add_seasonality(
            #     name="weekly",
            #     period=7,
            #     fourier_order=6,
            #     prior_scale=10,
            #     mode="additive",
            # )
        )

        if self.daily_seasonality in ["True", True]:
            self.m_gas.daily_seasonality = True
        else:
            self.m_gas.daily_seasonality = False

        if self.weekly_seasonality in ["True", True]:
            self.m_gas.weekly_seasonality = True
        else:
            self.m_gas.weekly_seasonality = False

        ##
        ## Set desired yearly seasonality
        ##

        if self.yearly_seasonality == 0:

            # 0 = no yearly seasonality
            self.m_gas.yearly_seasonality = False

        elif self.yearly_seasonality == 1:

            # 1 = default yearly seasonality
            self.m_gas.yearly_seasonality = True

        else:

            # 2 = custom yearly seasonality
            self.m_gas.add_seasonality(
                name="yearly",
                period=365,
                fourier_order=3,
                prior_scale=10,
                mode="additive",
            )

        # changepoint_prior_scale
        # This is probably the most impactful parameter. It determines the flexibility of the
        # trend, and in particular how much the trend changes at the trend changepoints. As
        # described in this documentation, if it is too small, the trend will be underfit and
        # variance that should have been modeled with trend changes will instead end up being
        # handled with the noise term. If it is too large, the trend will overfit and in the
        # most extreme case you can end up with the trend capturing yearly seasonality. The
        # default of 0.05 works for many time series, but this could be tuned; a range of
        # [0.001, 0.5] would likely be about right. Parameters like this (regularization
        # penalties; this is effectively a lasso penalty) are often tuned on a log scale.

        # seasonality_prior_scale
        # This parameter controls the flexibility of the seasonality. Similarly, a large value
        # allows the seasonality to fit large fluctuations, a small value shrinks the magnitude
        # of the seasonality. The default is 10., which applies basically no regularization.
        # That is because we very rarely see overfitting here (there’s inherent regularization
        # with the fact that it is being modeled with a truncated Fourier series, so it’s
        # essentially low-pass filtered). A reasonable range for tuning it would probably be
        # [0.01, 10]; when set to 0.01 you should find that the magnitude of seasonality is
        # forced to be very small. This likely also makes sense on a log scale, since it is
        # effectively an L2 penalty like in ridge regression.

        # holidays_prior_scale
        # This controls flexibility to fit holiday effects. Similar to seasonality_prior_scale,
        # it defaults to 10.0 which applies basically no regularization, since we usually have
        # multiple observations of holidays and can do a good job of estimating their effects.
        # This could also be tuned on a range of [0.01, 10] as with seasonality_prior_scale

        # self.m_gas.add_country_holidays(country_name="US")  # Holidays regressor

        if self.apply_weather_regression:
            if self.streamlit:
                with st.spinner("Adding weather regressor..."):
                    self.m_gas.add_regressor("temp")  # Temperature regressor
            else:
                self.m_gas.add_regressor("temp")  # Temperature regressor

            # Drop columns with nan values, so we don't get an error
            self.df_gas.dropna(subset=["temp"], inplace=True)

        # growth = logistic
        self.df_gas["cap"] = self.df_gas["y"].max()
        self.df_gas["floor"] = 0

        if self.streamlit:
            with st.beta_expander("Show Gas Usage Dataframe"):
                st.write(self.df_gas)

        if self.streamlit:
            with st.spinner("Fitting gas model..."):
                with suppress_stdout_stderr():
                    self.m_gas.fit(self.df_gas)
        else:
            with suppress_stdout_stderr():
                self.m_gas.fit(self.df_gas)

        # st.write(self.m_gas.train_holiday_names)
        # m_gas.history

        forecast_day_buffer = self.df_gas.iloc[-1]["ds"]
        delta = (
            pd.to_datetime(self.today_date).date()
            + datetime.timedelta(self.days_to_forecast)
        ) - pd.to_datetime(forecast_day_buffer).date()

        if self.streamlit:
            with st.spinner("Making future dataframe..."):
                self.future_gas = self.m_gas.make_future_dataframe(
                    periods=delta.days
                )  # Next 24x365 = 8760 hours
                # self.future_gas = self.m_gas.make_future_dataframe(periods = 7) # Next 24x365 = 8760 hours
        else:
            self.future_gas = self.m_gas.make_future_dataframe(
                periods=delta.days
            )  # Next 24x365 = 8760 hours

        # growth = logistic
        # self.future_gas['cap'] = df_gas['y'].max()
        # self.future_gas['floor'] = 0

        if self.apply_weather_regression:
            self.future_gas["temp"] = self.future_gas["ds"].apply(self.weather_temp)

        # Automatic FRI-SUN
        self.future_gas["weekend"] = self.future_gas["ds"].apply(
            self.is_weekend_or_weekday
        )
        self.future_gas["weekday"] = ~self.future_gas["ds"].apply(
            self.is_weekend_or_weekday
        )

        if self.streamlit:
            with st.spinner("Predicting the future..."):

                self.forecast_gas = self.m_gas.predict(self.future_gas)

        else:

            self.forecast_gas = self.m_gas.predict(self.future_gas)

    def weather_temp(self, ds):
        date = (pd.to_datetime(ds)).date()

        if str(date) not in self.df_weather.index:

            ## Get forecasted temp from database
            areas_row = self.projected_temp_df.loc[
                self.projected_temp_df["WeatherStationID"] == self.WeatherStationID
            ]

            areas_row = areas_row.loc[areas_row["Date"] == str(date)]

            if areas_row.empty:
                print(
                    "Ruh-roh.  No historical temperature for "
                    + str(self.WeatherStationID)
                    + " "
                    + str(date)
                )

            projected_temp = (areas_row["AvgTemp"]).values[0]
            return projected_temp

        else:

            return (self.df_weather[date:]["y"]).values[0]

        return 0


class site_specific_settings(object):
    def __init__(
        self,
        conn,
        site_id,
        days_of_usage,
        changepoint_prior_scale,
        changepoint_range,
        seasonality_mode,
        seasonality_prior_scale,
        daily_seasonality,
        weekly_seasonality,
        yearly_seasonality,
        apply_weather_regression,
    ):
        super(site_specific_settings, self).__init__()
        self.conn = conn
        self.site_id = site_id
        self.days_of_usage = days_of_usage
        self.changepoint_prior_scale = changepoint_prior_scale
        self.changepoint_range = changepoint_range
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.apply_weather_regression = apply_weather_regression

    def retrieve(self):

        site_params_df = pd.read_sql_query(
            "SELECT setting, value "
            + "FROM Site_Forecast_Settings "
            + "WHERE site_id = "
            + str(self.site_id),
            self.conn,
        )

        site_params_df.set_index("setting", inplace=True)

        try:
            if site_params_df.loc["days_of_usage", "value"]:
                self.days_of_usage = site_params_df.loc["days_of_usage", "value"]
        except:
            pass
        try:
            if site_params_df.loc["changepoint_prior_scale", "value"]:
                self.changepoint_prior_scale = site_params_df.loc[
                    "changepoint_prior_scale", "value"
                ]
        except:
            pass
        try:
            if site_params_df.loc["changepoint_range", "value"]:
                self.changepoint_range = site_params_df.loc[
                    "changepoint_range", "value"
                ]
        except:
            pass
        try:
            if site_params_df.loc["seasonality_mode", "value"]:
                self.seasonality_mode = site_params_df.loc["seasonality_mode", "value"]
        except:
            pass
        try:
            if site_params_df.loc["seasonality_prior_scale", "value"]:
                self.seasonality_prior_scale = site_params_df.loc[
                    "seasonality_prior_scale", "value"
                ]
        except:
            pass
        try:
            if site_params_df.loc["daily_seasonality", "value"]:
                self.daily_seasonality = site_params_df.loc[
                    "daily_seasonality", "value"
                ]
        except:
            pass
        try:
            if site_params_df.loc["weekly_seasonality", "value"]:
                self.weekly_seasonality = site_params_df.loc[
                    "weekly_seasonality", "value"
                ]
        except:
            pass
        try:
            if site_params_df.loc["yearly_seasonality", "value"]:
                self.yearly_seasonality = site_params_df.loc[
                    "yearly_seasonality", "value"
                ]
        except:
            pass
        try:
            if site_params_df.loc["apply_weather_regression", "value"]:
                self.apply_weather_regression = site_params_df.loc[
                    "apply_weather_regression", "value"
                ]
        except:
            pass

        return (
            self.days_of_usage,
            self.changepoint_prior_scale,
            self.changepoint_range,
            self.seasonality_mode,
            self.seasonality_prior_scale,
            self.daily_seasonality,
            self.weekly_seasonality,
            self.yearly_seasonality,
            self.apply_weather_regression,
        )

    def is_weekend_or_weekday(self, ds):
        date = pd.to_datetime(ds)
        return date.weekday() > 4 or date.weekday() < 1


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
