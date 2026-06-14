import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import linregress

from preprocess import get_merged_data, SEASON_COLORS, SEASON_ORDER

st.set_page_config(page_title="Energy & Weather — Spain", layout="wide")

PAL = {
    "load": "steelblue", "price": "mediumseagreen", "temp": "crimson",
    "humidity": "cadetblue", "weekday": "navy", "weekend": "firebrick",
    "fossil": "mediumpurple", "wind": "orange", "shade": "gray",
}

for k, v in [("year_filter", "All"), ("season_filter", "All")]:
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_data
def load_data():
    return get_merged_data()

def filtered():
    df = load_data()
    if st.session_state.year_filter != "All":
        df = df[df["year"] == st.session_state.year_filter]
    if st.session_state.season_filter != "All":
        df = df[df["season"] == st.session_state.season_filter]
    return df

df_full = load_data()
df = filtered()

#SIDEBAR NAVIGATION
st.sidebar.markdown("### Spain — Energy & Climate")
st.sidebar.caption("2015–2018 — 35,017 records")
st.sidebar.selectbox("Year", ["All", 2015, 2016, 2017, 2018], key="year_filter")
st.sidebar.selectbox("Season", ["All"] + SEASON_ORDER, key="season_filter")
st.sidebar.divider()
st.sidebar.header("Navigation")
st.sidebar.markdown(
    "- [Annotation](#annotation)\n"
    "- [Dashboard](#dashboard)\n"
    "- [Data & Pipeline](#data-pipeline)\n"
    "- [Dataset Overview](#dataset-overview)\n"
    "- [Time & Generation](#time-generation)\n"
    "- [Seasonal Analysis](#seasonal-analysis)\n"
    "- [Hypothesis Testing](#hypothesis-testing)\n"
    "- [Discussion](#discussion)"
)

#ANNOTATION
st.header("Annotation", anchor="annotation")
st.markdown("""
We are pleased to share our research on energy demand and generation, with data collected fresh from Spain! We focused our research on time-related dependencies, obtaining two hypothesis on energy use that we confirmed with statistical tests. 

The roles in the team were distributed as follows:
- **Nikita** - Data cleanup & broad analysis 
- **Alima** - Hypothesis formulation and statistical tests

Both of us made contributions to the API and Web interface, as well as to the vast collection of plots and graphs presented in our report.
""")
st.image("Gemini_Generated_Image_61g94l61g94l61g9.png", use_container_width=True)
st.divider()

#DASHBOARD
st.header("Dashboard", anchor="dashboard")
st.title("Energy Consumption & Weather — Spain (2015–2018)")
st.markdown("Hourly electricity generation, load, and pricing data for Spain merged with meteorological observations from five major Spanish cities, covering 2015 through 2018. Use the sidebar to filter by year or season.")

c = st.columns(4)
c[0].metric("Average Load", f"{df['total load actual'].mean():,.0f} MW")
c[1].metric("Average Temperature", f"{df['temp_c'].mean():.1f} °C")
c[2].metric("Average Price", f"{df['price actual'].mean():.1f} EUR/MWh")
c[3].metric("Average Humidity", f"{df['humidity'].mean():.0f}%")

c = st.columns(4)
c[0].metric("Peak Load", f"{df['total load actual'].max():,.0f} MW")
c[1].metric("Min Load", f"{df['total load actual'].min():,.0f} MW")
c[2].metric("Average Wind Speed", f"{df['wind_speed'].mean():.1f} m/s")
c[3].metric("Records", f"{len(df):,}")
st.divider()

#DATA & PIPELINE
st.header("Data & Pipeline", anchor="data-pipeline")
st.subheader("Data Cleanup")
st.markdown(
    "Timestamps are parsed to UTC. Fully-empty columns (100% NaN) are dropped. "
    "Rows with missing `total load actual` are removed. "
    "Weather duplicates are removed and observations from 5 cities "
    "are aggregated to hourly means for numeric fields and mode for text. "
    "Energy and weather datasets are inner-joined on timestamp. "
    "Temperature is converted from Kelvin to Celsius. "
    "Final dataset: 35,017 rows, 0 missing values."
)

st.subheader("Descriptive Statistics")
stat_cols = ["total load actual", "price actual", "temp_c", "humidity", "wind_speed"]
stats_df = df[stat_cols].agg(["mean", "median", "std", "min", "max"]).T.round(2)
stats_df.columns = ["Mean", "Median", "Std Dev", "Min", "Max"]
st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

st.subheader("Engineered Features")
tr_cols = [c for c in ["hour", "month", "season", "is_weekend", "is_peak_hour",
                        "temp_c", "humidity", "wind_speed"] if c in df.columns]
st.dataframe(df[tr_cols].head(8), use_container_width=True)
st.divider()

#DATASET OVERVIEW
st.header("Dataset Overview", anchor="dataset-overview")
st.subheader("Distributions")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x="total load actual", nbins=50, title="Total Load (MW)", color_discrete_sequence=[PAL["load"]])
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.histogram(df, x="price actual", nbins=50, title="Electricity Price (EUR/MWh)", color_discrete_sequence=[PAL["price"]])
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Temperature Timeline")
daily_temp = df.set_index("time_utc")["temp_c"].resample("D").mean().rolling(30).mean().reset_index()
fig3 = px.line(daily_temp, x="time_utc", y="temp_c", title="30-Day Rolling Average Temperature", color_discrete_sequence=[PAL["temp"]])
fig3.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Variable Spread")
col3, col4 = st.columns(2)
with col3:
    fig4 = px.box(df, y="temp_c", title="Temperature (°C)", color_discrete_sequence=["lightcoral"])
    st.plotly_chart(fig4, use_container_width=True)
with col4:
    fig5 = px.box(df, y="humidity", title="Humidity (%)", color_discrete_sequence=[PAL["humidity"]])
    st.plotly_chart(fig5, use_container_width=True)
st.divider()

#TIME & GENERATION
st.header("Time & Generation", anchor="time-generation")
st.subheader("Correlation Heatmap")
cols = ["total load actual", "price actual", "temp_c", "humidity", "wind_speed", "generation fossil gas", "generation wind onshore"]
cols = [c for c in cols if c in df_full.columns]
corr = df_full[cols].corr()
fig6 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
st.plotly_chart(fig6, use_container_width=True)

st.subheader("Price by Hour, Weekday, and Season")
cols = st.columns(4)
for i, season_name in enumerate(SEASON_ORDER):
    pivot = (df_full[df_full["season"] == season_name]
             .groupby(["day_of_week_num", "hour"])["price actual"]
             .mean().unstack(level="hour").reindex(index=range(7)))
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig7 = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd", title=season_name)
    fig7.update_layout(xaxis_title="Hour", yaxis_title="Day" if i==0 else "")
    cols[i].plotly_chart(fig7, use_container_width=True)

st.subheader("Diurnal Patterns")
hourly = df_full.groupby("hour")[["price actual", "total load actual", "generation fossil gas", "generation wind onshore"]].mean().reset_index()
hourly_melt = hourly.melt("hour")
fig8 = px.line(hourly_melt, x="hour", y="value", facet_col="variable", facet_col_wrap=2, markers=True, color="variable", title="Diurnal Patterns", height=600)
fig8.update_yaxes(matches=None)
fig8.update_layout(showlegend=False)
st.plotly_chart(fig8, use_container_width=True)
st.divider()

#SEASONAL ANALYSIS
st.header("Seasonal Analysis", anchor="seasonal-analysis")
st.subheader("Temperature vs Total Load by Season")
cols2 = st.columns(4)
for i, season_name in enumerate(SEASON_ORDER):
    sdf = df_full[df_full["season"] == season_name]
    fig9 = px.scatter(sdf, x="temp_c", y="total load actual", title=season_name, opacity=0.15)
    fig9.update_traces(marker=dict(color=SEASON_COLORS.get(season_name, "blue"), size=5))
    fig9.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Total Load (MW)" if i==0 else "")
    cols2[i].plotly_chart(fig9, use_container_width=True)

season_stats = df_full.groupby("season", observed=True).agg(
    mean_load=("total load actual", "mean"),
    mean_temp=("temp_c", "mean"),
    mean_price=("price actual", "mean"),
    hours=("total load actual", "count"),
).reindex(SEASON_ORDER)
st.dataframe(season_stats.style.format("{:,.0f}"), use_container_width=True)
st.divider()


#HYPOTHESIS TESTING
st.header("Hypothesis Testing", anchor="hypothesis-testing")
st.subheader("H1: Summer Temperature vs Load — Workdays vs Weekends")
summer = df_full[df_full["season"] == "Summer"].copy()
summer['Day Type'] = summer['is_weekend'].apply(lambda x: 'Weekend' if x == 1 else 'Workday')

wd = summer[summer["is_weekend"] == 0]
we = summer[summer["is_weekend"] == 1]

wd_reg = linregress(wd["temp_c"], wd["total load actual"])
we_reg = linregress(we["temp_c"], we["total load actual"])

fig10 = px.scatter(summer, x="temp_c", y="total load actual", color="Day Type", 
                   opacity=0.3, color_discrete_map={'Workday': PAL["weekday"], 'Weekend': PAL["weekend"]},
                   title="Summer Temperature vs Total Load")

xs = np.linspace(summer["temp_c"].min(), summer["temp_c"].max(), 100)
fig10.add_trace(go.Scatter(x=xs, y=wd_reg.intercept + wd_reg.slope * xs, mode='lines', name=f'Workday Trend (R²={wd_reg.rvalue**2:.3f})', line=dict(color='lightskyblue', width=5)))
fig10.add_trace(go.Scatter(x=xs, y=we_reg.intercept + we_reg.slope * xs, mode='lines', name=f'Weekend Trend (R²={we_reg.rvalue**2:.3f})', line=dict(color='lightcoral', width=5)))
st.plotly_chart(fig10, use_container_width=True)

t_h1, p_h1 = stats.ttest_ind(wd["total load actual"], we["total load actual"], equal_var=False)
st.markdown(
    f"**T-test (unequal variance):** Workday mean: `{wd['total load actual'].mean():,.0f} MW` | "
    f"Weekend mean: `{we['total load actual'].mean():,.0f} MW` | "
    f"**p-value:** `{p_h1:.2e}`. \n\n✅ Weekends have significantly lower load."
)

st.subheader("H2: Diurnal Peaks in Load and Price")
hourly_df = df_full.groupby("hour")[["total load actual", "price actual"]].mean().reset_index()

col5, col6 = st.columns(2)

fig11 = px.line(hourly_df, x="hour", y="total load actual", title="Average Load by Hour", markers=True)
fig11.update_traces(line_color=PAL["fossil"])
fig11.add_vrect(x0=8, x1=11, fillcolor="gray", opacity=0.2, annotation_text="Morning peak", line_width=0)
fig11.add_vrect(x0=18, x1=21, fillcolor="gray", opacity=0.2, annotation_text="Evening peak", line_width=0)
col5.plotly_chart(fig11, use_container_width=True)

fig12 = px.line(hourly_df, x="hour", y="price actual", title="Average Price by Hour", markers=True)
fig12.update_traces(line_color=PAL["wind"])
fig12.add_vrect(x0=8, x1=11, fillcolor="gray", opacity=0.2, annotation_text="Morning peak", line_width=0)
fig12.add_vrect(x0=18, x1=21, fillcolor="gray", opacity=0.2, annotation_text="Evening peak", line_width=0)
col6.plotly_chart(fig12, use_container_width=True)

peak_load = df_full[df_full["is_peak_hour"] == 1]["total load actual"]
off_load = df_full[df_full["is_peak_hour"] == 0]["total load actual"]
t_l, p_l = stats.ttest_ind(peak_load, off_load, equal_var=False)

peak_price = df_full[df_full["is_peak_hour"] == 1]["price actual"]
off_price = df_full[df_full["is_peak_hour"] == 0]["price actual"]
t_p, p_p = stats.ttest_ind(peak_price, off_price, equal_var=False)

st.markdown(
    f"**Load:** peak `{peak_load.mean():,.0f} MW` vs off-peak `{off_load.mean():,.0f} MW`, **p = {p_l:.2e}**. \n\n"
    f"**Price:** peak `{peak_price.mean():.2f} EUR/MWh` vs off-peak `{off_price.mean():.2f} EUR/MWh`, **p = {p_p:.2e}**. \n\n"
    f"✅ Both load and price are significantly higher during peak hours (p < 0.05)."
)
st.divider()

# DISCUSSION
st.header("Discussion", anchor="discussion")
st.markdown(
    "The data shows that electricity load in Spain depends a lot on temperature. "
    "Winter has the highest loads because of heating, and Summer loads go up when "
    "it gets hotter because of air conditioning. Spring and Autumn are lower overall."
)
st.markdown(
    "For the summer hypothesis, workdays had much higher load (around 29,900 MW on "
    "average) compared to weekends (about 27,400 MW). The temperature-load relationship "
    "was also stronger on workdays, with R² of 0.38 vs 0.22 on weekends. "
    "The t-test gave a very small p-value so the difference is real."
)
st.markdown(
    "The diurnal pattern is clear — load and price both spike during morning (8-11) "
    "and evening (18-21) hours. Peak load averages around 31,400 MW vs 27,300 MW "
    "off-peak, and prices follow the same pattern (63.1 vs 55.3 EUR/MWh). "
    "The p-values are basically zero so these peaks are definitely there."
)
st.markdown(
    "A few things to keep in mind: the data is for all of Spain so we can't see "
    "regional differences. Weather data comes from only 5 cities. Things like "
    "public holidays might affect the results. Also the data is from 2015-2018 "
    "so it doesn't reflect recent changes in Spain's energy mix."
)
