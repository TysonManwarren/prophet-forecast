clear
# forecast_type_id, progress_display, insert_into_db, days, CPS, Category-Specific Hyperparameters, Site-Specific Hyperparameters, daily_seasonality, weekly_seasonality, yearly_seasonality(0=none,1=on,2=custom)

progress_display=0
insert_into_db=1

#
# 30 days of use
# No Yearly Seasonality
#

python script_forecast.py 1 $progress_display $insert_into_db 30 .05 False False True True 0
python script_forecast.py 2 $progress_display $insert_into_db 30 .5 False False True True 0
python script_forecast.py 3 $progress_display $insert_into_db 30 .05 True False True True 0

#
# 60 days of use
# No Yearly Seasonality
#

python script_forecast.py 4 $progress_display $insert_into_db 60 .05 False False True True 0
python script_forecast.py 5 $progress_display $insert_into_db 60 .5 False False True True 0
python script_forecast.py 6 $progress_display $insert_into_db 60 .05 True False True True 0

#
# 365 days of use
# Custom Yearly Seasonality
#

python script_forecast.py 7 $progress_display $insert_into_db 365 .05 True False True True 2
python script_forecast.py 8 $progress_display $insert_into_db 365 .05 False False True True 2
python script_forecast.py 9 $progress_display $insert_into_db 365 .5 False False True True 2

#
# 30 days of use
# Custom Yearly Seasonality
#

python script_forecast.py 10 $progress_display $insert_into_db 30 .05 False False True True 2
python script_forecast.py 11 $progress_display $insert_into_db 30 .5 False False True True 2
python script_forecast.py 12 $progress_display $insert_into_db 30 .05 True False True True 2

#
# 30 days of use
# Site-specific settings
#

python script_forecast.py 13 $progress_display $insert_into_db 30 .05 False True True True 0
python script_forecast.py 14 $progress_display $insert_into_db 30 .5 False True True True 0
python script_forecast.py 15 $progress_display $insert_into_db 30 .05 True True True True 0

#
# 30 days of use
# Site-specific settings
# Custom Yearly Seasonality
#

python script_forecast.py 16 $progress_display $insert_into_db 30 .05 False True True True 2
python script_forecast.py 17 $progress_display $insert_into_db 30 .5 False True True True 2
python script_forecast.py 18 $progress_display $insert_into_db 30 .05 True True True True 2

#
# 60 and 365 days of use
# Site-specific settings
# Custom Yearly Seasonality
#

python script_forecast.py 19 $progress_display $insert_into_db 60 .05 False True True True 2
python script_forecast.py 20 $progress_display $insert_into_db 365 .05 False True True True 2