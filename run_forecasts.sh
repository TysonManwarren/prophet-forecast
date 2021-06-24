clear
# forecast_type_id, progress_display, insert_into_db, days, growth, CPS, Site-Specific Hyperparameters, daily_seasonality, weekly_seasonality, yearly_seasonality(0=none,1=on,2=custom)

progress_display=0
insert_into_db=1

##
## LINEAR GROWTH
##
##########################################################################

#
# 30 days of use
# No Yearly Seasonality
#

python script_forecast.py 1 $progress_display $insert_into_db linear 30 .05 False True True 0
python script_forecast.py 2 $progress_display $insert_into_db linear 30 .5 False True True 0

#
# 60 days of use
# No Yearly Seasonality
#

python script_forecast.py 4 $progress_display $insert_into_db linear 60 .05 False True True 0
python script_forecast.py 5 $progress_display $insert_into_db linear 60 .5 False True True 0

#
# 365 days of use
# Custom Yearly Seasonality
#

python script_forecast.py 8 $progress_display $insert_into_db linear 365 .05 False True True 2
python script_forecast.py 9 $progress_display $insert_into_db linear 365 .5 False True True 2

#
# 30 days of use
# Custom Yearly Seasonality
#

python script_forecast.py 10 $progress_display $insert_into_db linear 30 .05 False True True 2
python script_forecast.py 11 $progress_display $insert_into_db linear 30 .5 False True True 2

#
# 30 days of use
# Site-specific settings
#

python script_forecast.py 13 $progress_display $insert_into_db linear 30 .05 True True True 0
python script_forecast.py 14 $progress_display $insert_into_db linear 30 .5 True True True 0

#
# 30 days of use
# Site-specific settings
# Custom Yearly Seasonality
#

python script_forecast.py 16 $progress_display $insert_into_db linear 30 .05 True True True 2
python script_forecast.py 17 $progress_display $insert_into_db linear 30 .5 True True True 2

#
# 60 and 365 days of use
# Site-specific settings
# Custom Yearly Seasonality
#

python script_forecast.py 19 $progress_display $insert_into_db linear 60 .05 True True True 2
python script_forecast.py 20 $progress_display $insert_into_db linear 365 .05 True True True 2

##
## FLAT GROWTH
##
##########################################################################

#
# 30 days of use
# No Yearly Seasonality
#

python script_forecast.py 21 $progress_display $insert_into_db flat 30 .05 False True True 0
python script_forecast.py 22 $progress_display $insert_into_db flat 30 .5 False True True 0

#
# 60 days of use
# No Yearly Seasonality
#

python script_forecast.py 24 $progress_display $insert_into_db flat 60 .05 False True True 0
python script_forecast.py 25 $progress_display $insert_into_db flat 60 .5 False True True 0

#
# 365 days of use
# Custom Yearly Seasonality
#

python script_forecast.py 28 $progress_display $insert_into_db flat 365 .05 False True True 2
python script_forecast.py 29 $progress_display $insert_into_db flat 365 .5 False True True 2

#
# 30 days of use
# Custom Yearly Seasonality
#

python script_forecast.py 30 $progress_display $insert_into_db flat 30 .05 False True True 2
python script_forecast.py 31 $progress_display $insert_into_db flat 30 .5 False True True 2

#
# 30 days of use
# Site-specific settings
#

python script_forecast.py 32 $progress_display $insert_into_db flat 30 .05 True True True 0
python script_forecast.py 33 $progress_display $insert_into_db flat 30 .5 True True True 0

#
# 30 days of use
# Site-specific settings
# Custom Yearly Seasonality
#

python script_forecast.py 34 $progress_display $insert_into_db flat 30 .05 True True True 2
python script_forecast.py 35 $progress_display $insert_into_db flat 30 .5 True True True 2

#
# 60 and 365 days of use
# Site-specific settings
# Custom Yearly Seasonality
#

python script_forecast.py 36 $progress_display $insert_into_db flat 60 .05 True True True 2
python script_forecast.py 37 $progress_display $insert_into_db flat 365 .05 True True True 2