import pandas as pd
import pandas_access as mdb
from fbprophet import Prophet
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_squared_error

# plotting
import matplotlib.pyplot as plt

# settings
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = (16, 8)

# loading the dataset
# df = pd.read_csv("./neural_prophet/example_data/wp_log_peyton_manning.csv")
# print(f"The dataset contains {len(df)} observations.")
# df.head()
root_share = "/mnt/arminius"

MDB = root_share + "/DailyBalancing_be.mdb"
DRV = "{Microsoft Access Driver (*.mdb)}"
USR = ""
PWD = ""

df_gas_master = mdb.read_table(MDB, "tblUse")
# Remove data for sites not selected
df = df_gas_master[(df_gas_master.SiteID == 410)]

# Remove columns that we don't care about
df = df[["Date", "Use"]].dropna()

# Renaming Columns for simplicity
df.rename(columns={"Date": "ds", "Use": "y"}, inplace=True)

print(f"The dataset contains {len(df)} observations.")
df.head()


df.plot(x="ds", y="y", title="Log daily page views")

# getting the train/test split
test_length = 365
df_train = df.iloc[:-test_length]
df_test = df.iloc[-test_length:]

nprophet_model = NeuralProphet()
metrics = nprophet_model.fit(df_train, freq="D")
future_df = nprophet_model.make_future_dataframe(
    df_train, periods=3, n_historic_predictions=len(df_train)
)
preds_df_2 = nprophet_model.predict(future_df)

nprophet_model.plot(preds_df_2)
plt.savefig(fname="plot.png")

nprophet_model.plot_components(preds_df_2, residuals=True)
plt.savefig(fname="plot_components.png")

nprophet_model.plot_parameters()
plt.savefig(fname="plot_parameters.png")

print(preds_df_2)