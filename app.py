"""Streamlit demo for interactive 24-hour PJME historical forecasts."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PJME Demand Forecast Explorer", page_icon="⚡",
                   layout="wide")

REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY))

from src.data import chronological_split, load_hourly_csv
from src.models import TimesFMZeroShot, hour_of_week_profile, seasonal_naive


@st.cache_data(show_spinner=False)
def load_series(path: str) -> pd.Series:
    return load_hourly_csv(path)


@st.cache_resource(show_spinner="Loading TimesFM checkpoint…")
def load_timesfm(model_id: str, adapter_path: str):
    adapter = adapter_path if adapter_path else None
    return TimesFMZeroShot(context_length=512, horizon=24,
                           adapter=adapter, model_id=model_id)


st.title("PJME Demand Forecast Explorer")
st.caption("Compare a 24-hour forecast with historical PJME demand.")

data_path = REPOSITORY / "data" / "pjme_hourly.csv"
if not data_path.exists():
    st.error("PJME data is missing. Restore data/pjme_hourly.csv from the repository.")
    st.stop()

series = load_series(str(data_path))
splits = chronological_split(series)
origins = splits.test.index[:max(0, ((len(splits.test) - 24) // 24 + 1) * 24):24]
if not len(origins):
    st.error("The test partition does not contain a complete 24-hour forecast window.")
    st.stop()

model_labels = {
    "Weekly seasonal naive": "seasonal_naive",
    "Hour-of-week profile": "hour_of_week_profile",
    "TimesFM 2.5 zero-shot": "timesfm_zero_shot",
    "TimesFM 2.5 LoRA adapter": "timesfm_lora",
}
with st.sidebar:
    st.header("Forecast settings")
    selected_label = st.selectbox("Forecast method", list(model_labels))
    origin = st.select_slider(
        "Historical forecast origin",
        options=list(origins),
        value=origins[len(origins) // 2],
        format_func=lambda value: value.strftime("%Y-%m-%d %H:%M"),
        help="The following 24 historical hours are shown to score the forecast.",
    )
    st.caption("TimesFM methods download or load the base checkpoint on first use. "
               "The LoRA adapter is stored in this repository.")

position = int(series.index.get_loc(origin))
context = series.iloc[max(0, position - 512):position]
target_index = series.index[position:position + 24]
actual = series.iloc[position:position + 24].to_numpy(dtype=float)
if len(actual) != 24:
    st.error("The selected origin has no complete 24-hour actual window.")
    st.stop()

method = model_labels[selected_label]
if method == "seasonal_naive":
    forecast = seasonal_naive(context.to_numpy(dtype=float), 24)
elif method == "hour_of_week_profile":
    forecast = hour_of_week_profile(splits.train, context, target_index)
else:
    adapter = REPOSITORY / "outputs" / "timesfm-lora-pjme-t4" if method == "timesfm_lora" else None
    if adapter and not (adapter / "adapter_model.safetensors").exists():
        st.error("The PJME LoRA adapter weights are missing from outputs/timesfm-lora-pjme-t4.")
        st.stop()
    try:
        model = load_timesfm(
            "google/timesfm-2.5-200m-transformers",
            str(adapter) if adapter else "",
        )
        forecast = model.predict(context.to_numpy(dtype=np.float32))
    except (ImportError, RuntimeError, OSError) as exc:
        st.error("TimesFM could not load. Install the fine-tuning dependencies and "
                 "ensure the base model checkpoint is available.")
        st.exception(exc)
        st.stop()

errors = np.asarray(forecast, dtype=float) - actual
mae = float(np.mean(np.abs(errors)))
rmse = float(np.sqrt(np.mean(errors ** 2)))
bias = float(np.mean(errors))

left, middle, right = st.columns(3)
left.metric("24-hour MAE", f"{mae:,.0f} MW")
middle.metric("24-hour RMSE", f"{rmse:,.0f} MW")
right.metric("Mean forecast bias", f"{bias:+,.0f} MW")

forecast_frame = pd.DataFrame({
    "timestamp": target_index,
    "actual": actual,
    "forecast": forecast,
})
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(forecast_frame["timestamp"], forecast_frame["actual"],
        label="Actual demand", linewidth=2, marker="o", markersize=3)
ax.plot(forecast_frame["timestamp"], forecast_frame["forecast"],
        label="Forecast", linewidth=2, marker="o", markersize=3)
ax.set_ylabel("Demand (MW)")
ax.set_xlabel("Timestamp")
ax.set_title(f"{selected_label} · origin {origin:%Y-%m-%d %H:%M}")
ax.grid(alpha=0.2)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig, width="stretch",
          alt="Actual and forecasted PJME demand over the selected 24-hour window")
plt.close(fig)

with st.expander("Hourly forecast values"):
    shown = forecast_frame.copy()
    shown["absolute_error"] = np.abs(errors)
    st.dataframe(shown, width="stretch", hide_index=True)

st.info(
    "This is a historical backtest: the selected date comes from the project's test "
    "period, and the chart shows actual demand for the same 24 hours. Use validation "
    "data for model selection; this screen is for exploration, not live operational advice."
)

with st.expander("About the project"):
    st.write(
        "The project compares weekly seasonal naive, a training-fitted hour-of-week "
        "profile, zero-shot TimesFM 2.5, and a PJME LoRA adapter. The committed report "
        "uses 908 daily origins and includes paired block-bootstrap uncertainty, "
        "quarter metrics, and high-load error slices."
    )
    st.markdown("[Open the project README](https://github.com/gufrxn999/energy-demand-timesfm)")
