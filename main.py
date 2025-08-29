"""Streamlit dashboard app for tracking and visualizing Dreaming Spanish progress.

This application provides an interactive interface to:
- Load and display viewing data from the Dreaming Spanish API
- Visualize viewing patterns
- Track progress towards learning milestones (50, 150, 300, 600, 1000, 1500 hours)
- Generate predictions for future milestone achievements
- Display statistics like streaks, best days, and goal completion rates
- Export viewing data to CSV format
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils import (
    generate_future_predictions,
    get_best_days,
    get_initial_time,
    load_data,
    fetch_external_times_df,
    normalize_description,
    format_time,
)

# Set pandas option for future compatibility
pd.set_option("future.no_silent_downcasting", True)  # noqa: FBT003

UPCOMING_MILESTONES_CAP = 3
MILESTONES = [50, 150, 300, 600, 1000, 1500]
COLOUR_PALETTE = {
    "primary": "#2E86C1",  # Primary blue
    "7day_avg": "#FFA500",  # Orange
    "30day_avg": "#2ECC71",  # Green
    # Milestone colors - using an accessible and distinguishable gradient
    "50": "#FF6B6B",  # Coral red
    "150": "#4ECDC4",  # Turquoise
    "300": "#9B59B6",  # Purple
    "600": "#F1C40F",  # Yellow
    "1000": "#E67E22",  # Orange
    "1500": "#2ECC71",  # Green
}

st.set_page_config(page_title="Dreaming Spanish Time Tracker", layout="wide")

title = "Dreaming Spanish Stats"
subheader = "Analyze your viewing habits and predict your progress"

# Safely detect environment from URL (fallback to local when unavailable)
try:
    current_url = st.context.url  # type: ignore[attr-defined]
except Exception:
    current_url = None

if current_url and current_url.startswith("https://ds-stats-dev."):
    title += " - :orange[Dev Build]"
elif not (current_url and current_url.startswith("https://ds-stats.")):
    title += " - :violet[Local Build]"

st.title(title)
st.subheader(subheader)

# Show warning for dev build
if current_url and current_url.startswith("https://ds-stats-dev."):
    st.warning(
        "You are viewing the development version of the application, meaning "
        "that it may not be fully functional, may contain bugs, or may be "
        "using experimental features.\n Resort to the production version if "
        "you encounter any issues.",
    )

button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
with button_col1:
    st.link_button(
        "☁️ Official Progress",
        "https://www.dreamingspanish.com/progress",
        use_container_width=True,
    )

with button_col2:
    st.link_button(
        "🪲 Report Issue",
        "http://github.com/HarryPeach/dreamingspanishstats/issues",
        use_container_width=True,
    )

with button_col3:
    st.link_button(
        "📖 Source Code",
        "http://github.com/HarryPeach/dreamingspanishstats",
        use_container_width=True,
    )

# Load default token from file (overrideable by input)
DEFAULT_TOKEN_FILE = Path(r"C:\Users\drobchak\Desktop\Espanol\ds_auth_token.txt")
default_token = ""
try:
    if DEFAULT_TOKEN_FILE.exists():
        default_token = DEFAULT_TOKEN_FILE.read_text(encoding="utf-8").strip()
except Exception:
    default_token = ""

# Add token input and buttons in an aligned row
st.write("")  # Add some spacing
col1, col2 = st.columns([4, 1])
with col1:
    token = st.text_input(
        "Enter your bearer token:",
        type="password",
        key="token_input",
        label_visibility="collapsed",
        value=default_token,  # prefill from file by default
    )
with col2:
    go_button = st.button("Go", type="primary", use_container_width=True)

if not token:
    st.warning("Please enter your bearer token above to fetch data")
    # Load and display README
    try:
        with Path("bearer_how_to.md").open() as file:
            readme_content = file.read()
            if readme_content:
                st.markdown(readme_content, unsafe_allow_html=True)
    except FileNotFoundError:
        pass
    st.stop()

# Load data when token is provided and button is clicked
if "data" not in st.session_state or go_button:
    with st.spinner("Fetching data..."):
        data = load_data(token)
        if data is None:
            st.error(
                "Failed to fetch data from the DreamingSpanish API."
                "Please check your bearer token, ensuring it doesn't contain "
                "anything extra such as 'token:' at the beginning.",
            )
            st.stop()
        st.session_state.data = data

result = st.session_state.data
df = result.df
initial_time = get_initial_time(token) or 0
goals_reached = result.goals_reached
total_days = result.total_days
current_goal_streak = result.current_goal_streak
longest_goal_streak = result.longest_goal_streak

df = result.df.rename(columns={"timeSeconds": "seconds"})

# Calculate cumulative seconds and streak
df["cumulative_seconds"] = df["seconds"].cumsum() + initial_time
df["cumulative_minutes"] = df["cumulative_seconds"] / 60
df["cumulative_hours"] = df["cumulative_minutes"] / 60
df["streak"] = (df["seconds"] > 0).astype(int)

# Calculate current streak
df["streak_group"] = (df["streak"] != df["streak"].shift()).cumsum()
df["current_streak"] = df.groupby("streak_group")["streak"].cumsum()
current_streak = df["current_streak"].iloc[-1] if df["streak"].iloc[-1] == 1 else 0

# Calculate all-time longest streak
streak_lengths = df[df["streak"] == 1].groupby("streak_group").size()
longest_streak = streak_lengths.max() if not streak_lengths.empty else 0

# Calculate moving averages
df["7day_avg"] = df["seconds"].rolling(7, min_periods=1).mean()
df["30day_avg"] = df["seconds"].rolling(30, min_periods=1).mean()

avg_seconds_per_day = df["seconds"].mean()

with st.container(border=True):
    st.subheader("Basic Stats")

    # Current stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if initial_time > 0:
            st.metric(
                "Total Hours Watched",
                f"{df['cumulative_hours'].iloc[-1]:.1f}",
                f"including {initial_time / 60:.0f} min initial time",
            )
        else:
            st.metric("Total Hours Watched", f"{df['cumulative_hours'].iloc[-1]:.1f}")
    with col2:
        st.metric("Average Minutes/Day", f"{(avg_seconds_per_day / 60):.1f}")
    with col3:
        st.metric("Current Streak", f"{current_streak} days")
    with col4:
        st.metric("Longest Streak", f"{longest_streak} days")


with st.container(border=True):
    st.subheader("Projected Growth")

    # Calculate target milestone
    current_hours = df["cumulative_hours"].iloc[-1]
    upcoming_milestones = [m for m in MILESTONES if m > current_hours][:3]
    target_milestone = (
        upcoming_milestones[2]
        if len(upcoming_milestones) >= UPCOMING_MILESTONES_CAP
        else MILESTONES[-1]
    )

    # Calculate current moving averages for predictions
    current_7day_avg = df["7day_avg"].iloc[-1]
    current_30day_avg = df["30day_avg"].iloc[-1]

    # Generate predictions up to target milestone
    predicted_df = generate_future_predictions(
        df,
        avg_seconds_per_day,
        target_milestone,
    )
    predicted_df_7day = generate_future_predictions(
        df,
        current_7day_avg,
        target_milestone,
    )
    predicted_df_30day = generate_future_predictions(
        df,
        current_30day_avg,
        target_milestone,
    )

    # Create milestone prediction visualization
    fig_prediction = go.Figure()

    # Add historical data
    fig_prediction.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cumulative_hours"],
            name="Historical Data",
            line={"color": COLOUR_PALETTE["primary"]},
            mode="lines+markers",
        ),
    )

    # Add predicted data - Overall Average
    fig_prediction.add_trace(
        go.Scatter(
            x=predicted_df["date"],
            y=predicted_df["cumulative_hours"],
            name="Predicted (Overall Avg)",
            line={"color": f"{COLOUR_PALETTE['primary']}", "dash": "dash"},
            mode="lines",
            opacity=0.5,
        ),
    )

    # Add predicted data - 7-Day Average
    fig_prediction.add_trace(
        go.Scatter(
            x=predicted_df_7day["date"],
            y=predicted_df_7day["cumulative_hours"],
            name="Predicted (7-Day Avg)",
            line={"color": COLOUR_PALETTE["7day_avg"], "dash": "dot"},
            mode="lines",
            opacity=0.5,
        ),
    )

    # Add predicted data - 30-Day Average
    fig_prediction.add_trace(
        go.Scatter(
            x=predicted_df_30day["date"],
            y=predicted_df_30day["cumulative_hours"],
            name="Predicted (30-Day Avg)",
            line={"color": COLOUR_PALETTE["30day_avg"], "dash": "dot"},
            mode="lines",
            opacity=0.5,
        ),
    )

    for milestone in MILESTONES:
        color = COLOUR_PALETTE[str(milestone)]
        if milestone <= df["cumulative_hours"].max():
            milestone_date = df[df["cumulative_hours"] >= milestone]["date"].iloc[0]
        elif milestone <= predicted_df["cumulative_hours"].max():
            milestone_date = predicted_df[
                predicted_df["cumulative_hours"] >= milestone
            ]["date"].iloc[0]
        else:
            continue

        fig_prediction.add_shape(
            type="line",
            x0=df["date"].min(),
            x1=milestone_date,
            y0=milestone,
            y1=milestone,
            line={"color": color, "dash": "dash", "width": 1},
        )

        fig_prediction.add_annotation(
            x=df["date"].min(),
            y=milestone,
            text=f"{milestone} Hours",
            showarrow=False,
            xshift=-5,
            xanchor="right",
            font={"color": color},
        )

        fig_prediction.add_annotation(
            x=milestone_date,
            y=milestone,
            text=milestone_date.strftime("%Y-%m-%d"),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor=color,
            font={"color": color, "size": 10},
            xanchor="left",
            yanchor="bottom",
        )

    # Find the next 3 upcoming milestones and their dates
    current_hours = df["cumulative_hours"].iloc[-1]
    upcoming_milestones = [m for m in MILESTONES if m > current_hours][:3]
    y_axis_max = (
        upcoming_milestones[2]
        if len(upcoming_milestones) >= UPCOMING_MILESTONES_CAP
        else MILESTONES[-1]
    )

    # Get the date for the third upcoming milestone (or last milestone if < 3 remain)
    if len(upcoming_milestones) > 0:
        target_milestone = upcoming_milestones[min(2, len(upcoming_milestones) - 1)]
        milestone_data = predicted_df[
            predicted_df["cumulative_hours"] >= target_milestone
        ]
        if len(milestone_data) > 0:
            x_axis_max_date = milestone_data["date"].iloc[0]
        else:
            x_axis_max_date = predicted_df["date"].max()
    else:
        x_axis_max_date = predicted_df["date"].max()

    fig_prediction.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Hours",
        showlegend=True,
        height=600,
        margin={"l": 20, "r": 20, "t": 10, "b": 0},
        yaxis={
            "autorange": True,
        },
        xaxis={
            "autorange": True,
        },
    )

    st.plotly_chart(fig_prediction, use_container_width=True)

with st.container(border=True):
    st.subheader("Additional Graphs")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Moving Averages",
            "Daily Breakdown",
            "Yearly Heatmap",
            "Monthly Breakdown",
            "Days of Week",
        ],
    )

    with tab1:
        # Moving averages visualization
        moving_avg_fig = go.Figure()

        # Calculate cumulative average (running mean)
        df["cumulative_avg"] = df["seconds"].expanding().mean()

        moving_avg_fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["seconds"] / 3600,
                name="Daily Hours",
                mode="markers",
                marker={"size": 6},
            ),
        )

        moving_avg_fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["7day_avg"] / 3600,
                name="7-day Average",
                line={"color": COLOUR_PALETTE["7day_avg"]},
            ),
        )

        moving_avg_fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["30day_avg"] / 3600,
                name="30-day Average",
                line={"color": COLOUR_PALETTE["30day_avg"]},
            ),
        )

        moving_avg_fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["cumulative_avg"] / 3600,
                name="Overall Average",
                line={"color": COLOUR_PALETTE["primary"], "dash": "dash"},
            ),
        )

        moving_avg_fig.update_layout(
            title="Daily Hours with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Hours",
            height=450,
        )

        moving_avg_fig.update_yaxes(title="Hours Watched")

        st.plotly_chart(moving_avg_fig, use_container_width=True)

    with tab2:
        # Daily breakdown
        daily_fig = go.Figure()

        daily_fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["seconds"] / 60,  # Convert to minutes
                name="Daily Minutes",
            ),
        )

        daily_fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=[avg_seconds_per_day / 60] * len(df),  # Convert to minutes
                name="Overall Average",
                line={"color": COLOUR_PALETTE["primary"], "dash": "dash"},
            ),
        )

        daily_fig.update_layout(
            title="Daily Minutes Watched",
            xaxis_title="Date",
            yaxis_title="Minutes",
            height=450,
        )

        daily_fig.update_yaxes(dtick=15, title="Minutes Watched", ticklabelstep=2)
        st.plotly_chart(daily_fig, use_container_width=True)

    with tab3:
        # Create a complete year date range
        today = pd.Timestamp.now()
        year_start = pd.Timestamp(today.year, 1, 1)
        year_end = pd.Timestamp(today.year, 12, 31)
        all_dates = pd.date_range(year_start, year_end, freq="D")

        # Create a DataFrame with all dates
        full_year_df = pd.DataFrame({"date": all_dates})
        full_year_df["seconds"] = 0

        full_year_df = full_year_df.merge(
            df[["date", "seconds"]],
            on="date",
            how="left",
        )
        full_year_df["seconds"] = full_year_df["seconds_y"].fillna(0)

        # Calculate week and weekday using isocalendar
        isocalendar_df = full_year_df["date"].dt.isocalendar()
        full_year_df["weekday"] = isocalendar_df["day"]

        # Handle week numbers correctly
        full_year_df["week"] = isocalendar_df["week"]
        # Adjust week numbers for consistency
        mask = (full_year_df["date"].dt.month == 12) & (full_year_df["week"] <= 1)  # noqa: PLR2004
        full_year_df.loc[mask, "week"] = full_year_df.loc[mask, "week"] + 52
        mask = (full_year_df["date"].dt.month == 1) & (full_year_df["week"] >= 52)  # noqa: PLR2004
        full_year_df.loc[mask, "week"] = full_year_df.loc[mask, "week"] - 52

        # Rest of the heatmap code remains the same
        heatmap_fig = go.Figure()

        heatmap_fig.add_trace(
            go.Heatmap(
                x=full_year_df["week"],
                y=full_year_df["weekday"],
                z=full_year_df["seconds"] / 60,  # Convert to minutes
                colorscale=[
                    [0, "rgba(227,224,227,.5)"],  # Grey for zeros/future
                    [0.001, "rgb(243,231,154)"],
                    [0.5, "rgb(246,90,109)"],
                    [1, "rgb(126,29,103)"],
                ],
                showscale=True,
                colorbar={"title": "Minutes"},
                hoverongaps=False,
                hovertemplate="Date: %{customdata}<br>Minutes: %{z:.1f}<extra></extra>",
                customdata=full_year_df["date"].dt.strftime("%Y-%m-%d"),
                xgap=3,  # Add 3 pixels gap between columns
                ygap=3,  # Add 3 pixels gap between rows
            ),
        )

        # Update layout for GitHub-style appearance
        heatmap_fig.update_layout(
            title="Yearly Activity Heatmap",
            xaxis_title="Week",
            yaxis_title="Day of Week",
            height=300,
            yaxis={
                "ticktext": ["", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                "tickvals": [0, 1, 2, 3, 4, 5, 6, 7],
                "showgrid": False,
                "autorange": "reversed",  # This ensures Mon-Sun order
            },
            xaxis={
                "showgrid": False,
                "dtick": 1,  # Show all week numbers
                "range": [0.5, 53.5],  # Fix the range to show all weeks
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(heatmap_fig, use_container_width=True)

    with tab4:
        # Monthly breakdown
        df["month_year"] = df["date"].dt.to_period("M")

        last_12_months = sorted(df["month_year"].unique(), reverse=True)[:12][::-1]

        monthly_data = []
        today = datetime.now(tz=UTC).date()

        for month_period in last_12_months:
            month_df = df[df["month_year"] == month_period]

            days_practiced = month_df[month_df["seconds"] > 0]["date"].nunique()
            days_target_met = month_df["goalReached"].sum()

            if month_period.year == today.year and month_period.month == today.month:
                days_in_month = month_df["date"].nunique()
            else:
                days_in_month = month_period.days_in_month

            monthly_data.append(
                {
                    "month": month_period.strftime("%Y-%m"),
                    "days_practiced": days_practiced,
                    "days_target_met": days_target_met,
                    "days_in_month": days_in_month,
                },
            )

        monthly_df = pd.DataFrame(monthly_data)

        # Create grouped bar chart
        monthly_fig = go.Figure()

        monthly_fig.add_trace(
            go.Bar(
                x=monthly_df["month"],
                y=monthly_df["days_target_met"],
                name="Days Target Met",
                marker_color=COLOUR_PALETTE["7day_avg"],
            ),
        )

        monthly_fig.add_trace(
            go.Bar(
                x=monthly_df["month"],
                y=monthly_df["days_practiced"],
                name="Days Practiced (> 0 mins)",
                marker_color=COLOUR_PALETTE["primary"],
            ),
        )

        monthly_fig.add_trace(
            go.Bar(
                x=monthly_df["month"],
                y=monthly_df["days_in_month"],
                name="Tracked Days in Month",
                marker_color=COLOUR_PALETTE["30day_avg"],
            ),
        )

        monthly_fig.update_layout(
            barmode="group",
            title="Monthly Breakdown of Practice and Goals",
            xaxis_title="Month",
            yaxis_title="Number of Days",
            height=450,
            legend_title="Metric",
        )

        st.plotly_chart(monthly_fig, use_container_width=True)

    with tab5:
        # Days of week breakdown
        df["day_of_week"] = df["date"].dt.day_name()
        daily_avg_df = (
            df.groupby("day_of_week")["seconds"]
            .mean()
            .reindex(
                [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ],
            )
            .reset_index()
        )
        daily_avg_df["minutes"] = daily_avg_df["seconds"] / 60

        days_of_week_fig = px.bar(
            daily_avg_df,
            x="day_of_week",
            y="minutes",
            title="Average Minutes Watched per Day of Week",
            labels={"day_of_week": "Day of Week", "minutes": "Average Minutes Watched"},
            color_discrete_sequence=[COLOUR_PALETTE["primary"]],
        )

        days_of_week_fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Minutes",
            height=450,
        )
        st.plotly_chart(days_of_week_fig, use_container_width=True)

with st.container(border=True):
    st.subheader("Sources")

    # Prefer data fetched during load_data; fall back to direct fetch
    external_df = getattr(result, "external_df", None)
    if external_df is None or external_df.empty:
        external_df = fetch_external_times_df(token) or pd.DataFrame()

    if external_df.empty:
        st.info("No external sources data found.")
    else:
        ext = external_df.copy()

        # Exclude onboarding initial time and blank descriptions for leaderboard rows
        ext = ext[(ext.get("type", "") != "initial") & ext["description"].fillna("").ne("")]
        # Normalize and exclude “Dreaming Spanish” self-entries from leaderboard sources
        ext["norm_desc"] = ext["description"].fillna("").apply(normalize_description)
        ext = ext[ext["norm_desc"] != "dreaming spanish"]

        # Compute Dreaming Spanish (DS) time from day totals minus external sums (do not filter out DS here)
        daily_totals = df[["date", "seconds"]].rename(columns={"seconds": "total_seconds"}).set_index("date")
        ext_calc = external_df.copy()
        # Sum all external seconds by date (keep everything, including entries described as “Dreaming Spanish”)
        ext_by_date = (
            ext_calc.assign(date=pd.to_datetime(ext_calc["date"]))
            .groupby("date")["timeSeconds"]
            .sum()
            .to_frame("ext_seconds")
        )
        merged_daily = daily_totals.join(ext_by_date, how="left").fillna({"ext_seconds": 0})
        merged_daily["ds_seconds"] = (merged_daily["total_seconds"] - merged_daily["ext_seconds"]).clip(lower=0)
        ds_overall_seconds = float(merged_daily["ds_seconds"].sum())
        ds_monthly_seconds = merged_daily["ds_seconds"].groupby(merged_daily.index.to_period("M")).sum()

        if ext.empty and ds_overall_seconds == 0:
            st.info("No external sources after filtering initial time and Dreaming Spanish.")
        else:
            ext["month_year"] = ext["date"].dt.to_period("M")

            # Overall leaderboard (external sources)
            overall = (
                ext.groupby("norm_desc")
                .agg(seconds=("timeSeconds", "sum"), sessions=("id", "count"))
                .reset_index()
                .sort_values("seconds", ascending=False)
            )
            # Append Dreaming Spanish computed time (no sessions)
            if ds_overall_seconds > 0:
                ds_row = pd.DataFrame(
                    {"norm_desc": ["dreaming spanish"], "seconds": [ds_overall_seconds], "sessions": [None]},
                )
                overall = pd.concat([overall, ds_row], ignore_index=True)
                overall = overall.sort_values("seconds", ascending=False)

            # Prepare display columns
            overall["Source"] = overall["norm_desc"].apply(lambda s: " ".join(w.capitalize() for w in s.split()))
            overall["Time"] = (overall["seconds"] / 3600).apply(format_time)
            overall["Sessions"] = overall["sessions"].apply(lambda x: "" if pd.isna(x) else str(int(x))).astype("string")
            overall = overall.reset_index(drop=True)
            overall["#"] = range(1, len(overall) + 1)
            overall_display = overall[["#", "Source", "Time", "Sessions"]]

            # Monthly tabs: include only months that have data (external >0 or DS >0)
            ext_months = (
                set(
                    ext.groupby("month_year")["timeSeconds"]
                    .sum()
                    .pipe(lambda s: s[s > 0].index)
                )
                if not ext.empty
                else set()
            )
            ds_months = set(ds_monthly_seconds[ds_monthly_seconds > 0].index)
            months = sorted(list(ext_months.union(ds_months)))  # chronological (oldest -> newest)
            month_labels = [m.strftime("%b %Y") for m in months]

            # Years list (union of years present in external and DS monthly)
            # External years with non-zero total seconds
            ext_years = (
                set(
                    ext.groupby(ext["date"].dt.year)["timeSeconds"]
                    .sum()
                    .pipe(lambda s: s[s > 0].index)
                )
                if not ext.empty
                else set()
            )
            ds_years = set(pd.Index(ds_monthly_seconds[ds_monthly_seconds > 0].index).year)
            years = sorted(list(ext_years.union(ds_years)))

            # Single tab bar: Overall + Years + Months (no separator)
            tab_specs: list[tuple[str, object, str]] = [("overall", None, "Overall")]
            if years:
                tab_specs += [("year", y, str(y)) for y in years]
            tab_specs += [("month", m, m.strftime("%b %Y")) for m in months]
            tabs_combined = st.tabs([label for _, _, label in tab_specs])

            for t_idx, (kind, value, _) in enumerate(tab_specs):
                with tabs_combined[t_idx]:
                    if kind == "overall":
                        st.dataframe(overall_display.set_index("#"), use_container_width=True)
                    elif kind == "month":
                        month = value
                        month_rows = ext[ext["month_year"] == month]
                        mgrp = (
                            month_rows.groupby("norm_desc")
                            .agg(seconds=("timeSeconds", "sum"), sessions=("id", "count"))
                            .reset_index()
                        )
                        ds_month_seconds = float(ds_monthly_seconds.get(month, 0.0))
                        if ds_month_seconds > 0:
                            ds_month_row = pd.DataFrame(
                                {"norm_desc": ["dreaming spanish"], "seconds": [ds_month_seconds], "sessions": [None]},
                            )
                            mgrp = pd.concat([mgrp, ds_month_row], ignore_index=True)
                        if mgrp.empty:
                            st.write("No data for this month.")
                        else:
                            mgrp = mgrp.sort_values("seconds", ascending=False).reset_index(drop=True)
                            mgrp["Source"] = mgrp["norm_desc"].apply(lambda s: " ".join(w.capitalize() for w in s.split()))
                            mgrp["Time"] = (mgrp["seconds"] / 3600).apply(format_time)
                            mgrp["Sessions"] = mgrp["sessions"].apply(lambda x: "" if pd.isna(x) else str(int(x))).astype("string")
                            mgrp["#"] = range(1, len(mgrp) + 1)
                            mdisplay = mgrp[["#", "Source", "Time", "Sessions"]]
                            st.dataframe(mdisplay.set_index("#"), use_container_width=True)
                    elif kind == "year":
                        year = int(value)
                        ydf = ext[ext["date"].dt.year == year]
                        ygrp = (
                            ydf.groupby("norm_desc")
                            .agg(seconds=("timeSeconds", "sum"), sessions=("id", "count"))
                            .reset_index()
                        )
                        ds_year_seconds = float(
                            ds_monthly_seconds[pd.Index(ds_monthly_seconds.index).year == year].sum(),
                        )
                        if ds_year_seconds > 0:
                            ygrp = pd.concat(
                                [
                                    ygrp,
                                    pd.DataFrame(
                                        {"norm_desc": ["dreaming spanish"], "seconds": [ds_year_seconds], "sessions": [None]},
                                    ),
                                ],
                                ignore_index=True,
                            )
                        if ygrp.empty:
                            st.write("No data for this year.")
                        else:
                            ygrp = ygrp.sort_values("seconds", ascending=False).reset_index(drop=True)
                            ygrp["Source"] = ygrp["norm_desc"].apply(lambda s: " ".join(w.capitalize() for w in s.split()))
                            ygrp["Time"] = (ygrp["seconds"] / 3600).apply(format_time)
                            ygrp["Sessions"] = ygrp["sessions"].apply(lambda x: "" if pd.isna(x) else str(int(x))).astype("string")
                            ygrp["#"] = range(1, len(ygrp) + 1)
                            ydisplay = ygrp[["#", "Source", "Time", "Sessions"]]
                            st.dataframe(ydisplay.set_index("#"), use_container_width=True)

            st.markdown("#### Daily Cumulative Progress (Top 20 Sources)")
            # Build full daily date range from overall DF span
            day_min = df["date"].min().normalize()
            day_max = df["date"].max().normalize()
            full_days = pd.date_range(day_min, day_max, freq="D")

            # External sources per day per normalized description (exclude DS self-entries already filtered)
            if ext.empty:
                ext_pivot = pd.DataFrame(index=full_days)
            else:
                ext_day = (
                    ext.assign(date=ext["date"].dt.normalize())
                    .groupby(["date", "norm_desc"])["timeSeconds"]
                    .sum()
                    .unstack(fill_value=0.0)
                )
                ext_pivot = ext_day.reindex(full_days, fill_value=0.0)

            # Dreaming Spanish seconds per day = total - external (computed earlier in merged_daily)
            ds_series = (
                merged_daily["ds_seconds"]
                .reindex(full_days, fill_value=0.0)
                .rename("dreaming spanish")
            )

            # Combine into a wide daily matrix (days x sources)
            daily_sources_wide = ext_pivot.join(ds_series, how="left").fillna(0.0)

            # Compute cumulative seconds per source over days
            daily_cum_wide = daily_sources_wide.cumsum()

            # Prepare long-form data for plotting
            daily_cum_long = (
                daily_cum_wide.reset_index(names="date")
                .melt(id_vars=["date"], var_name="norm_desc", value_name="cum_seconds")
            )
            daily_cum_long["cum_hours"] = daily_cum_long["cum_seconds"] / 3600.0
            daily_cum_long["Source"] = daily_cum_long["norm_desc"].apply(
                lambda s: " ".join(w.capitalize() for w in s.split()),
            )

            # Top 20 sources by total cumulative hours at the end
            totals = daily_cum_wide.iloc[-1]
            top_sources = totals.sort_values(ascending=False).head(20).index.tolist()
            source_order = [" ".join(w.capitalize() for w in s.split()) for s in top_sources]

            chart_df = daily_cum_long[daily_cum_long["norm_desc"].isin(top_sources)].copy()
            chart_df = chart_df.sort_values(["Source", "date"])

            # Tabs: Cumulative vs 30-day Average
            tab_src_cum, tab_src_avg = st.tabs(["Cumulative", "30-day Average"])

            with tab_src_cum:
                fig_sources = px.line(
                    chart_df,
                    x="date",
                    y="cum_hours",
                    color="Source",
                    category_orders={"Source": source_order},
                    labels={"date": "Date", "cum_hours": "Cumulative Hours"},
                    markers=False,
                    title=None,
                )
                fig_sources.update_traces(line={"width": 3})
                fig_sources.update_layout(
                    height=450,
                    xaxis_title="Date",
                    yaxis_title="Cumulative Hours",
                    legend_title="Source",
                    margin={"l": 10, "r": 10, "t": 10, "b": 0},
                )
                fig_sources.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_sources, use_container_width=True)

            with tab_src_avg:
                # 30-day rolling average hours/day per source (top 20 only)
                hours_wide = daily_sources_wide / 3600.0
                cols = [c for c in top_sources if c in hours_wide.columns]
                if cols:
                    ma30_wide = hours_wide[cols].rolling(30, min_periods=1).mean()
                    ma30_long = (
                        ma30_wide.reset_index(names="date")
                        .melt(id_vars=["date"], var_name="norm_desc", value_name="hours")
                    )
                    ma30_long["Source"] = ma30_long["norm_desc"].apply(
                        lambda s: " ".join(w.capitalize() for w in s.split()),
                    )
                    fig_sources_avg = px.line(
                        ma30_long,
                        x="date",
                        y="hours",
                        color="Source",
                        category_orders={"Source": source_order},
                        labels={"date": "Date", "hours": "Avg Hours/Day"},
                        title=None,
                    )
                    fig_sources_avg.update_traces(line={"width": 3})
                    fig_sources_avg.update_layout(
                        height=450,
                        xaxis_title="Date",
                        yaxis_title="Avg Hours/Day",
                        legend_title="Source",
                        margin={"l": 10, "r": 10, "t": 10, "b": 0},
                    )
                    fig_sources_avg.update_yaxes(rangemode="tozero")
                    st.plotly_chart(fig_sources_avg, use_container_width=True)
                else:
                    st.write("No data to compute 30-day averages.")

            # Build daily classification from the raw external data (ext_calc) to include talking and all descriptions
            ext_all = ext_calc.copy()
            if not ext_all.empty:
                ext_all["date"] = pd.to_datetime(ext_all["date"]).dt.normalize()
                ext_all["type"] = ext_all.get("type", "").astype("string").str.lower().fillna("")
                ext_all["norm_desc"] = ext_all["description"].fillna("").apply(normalize_description)
            else:
                ext_all = pd.DataFrame(columns=["date", "type", "norm_desc", "timeSeconds"])

            # Learner content sources list (mirrors count_external_hours.py)
            LEARNER_CONTENT_SOURCES = {
                "dreaming spanish", "hoy hablamos", "que pasa",
                "duolingo podcast", "languatalk", "babbel", "the spanish on the road",
                "spanish language coach", "erre que ele", "hola spanish", "easy spanish",
                "handyspanish", "fluent spanish express", "la lengua de babel",
                "spanish for the camino", "profe de flele", "espanolistos", "charlas hispanas",
                "andalusian spanish to go", "español con juan", "si comprendo", "español coloquial y tal",
                "spanish boost", "<old podcasts>", "<old watching>", "dreaming youtube", "español automático",
            }

            # ========================
            # Content Type Leaderboard
            # ========================
            st.markdown("#### Content Type Leaderboard")
            tab_specs_ct: list[tuple[str, object, str]] = [("overall", None, "Overall")]
            if years:
                tab_specs_ct += [("year", y, str(y)) for y in years]
            tab_specs_ct += [("month", m, m.strftime("%b %Y")) for m in months]
            tabs_ct = st.tabs([label for _, _, label in tab_specs_ct])

            def compute_content_type_seconds(df_ext: pd.DataFrame) -> dict[str, float]:
                if df_ext.empty:
                    return {"Content for Learners": 0.0, "Native Content": 0.0, "Talking": 0.0}
                talking_sec = float(df_ext[df_ext["type"] == "talking"]["timeSeconds"].sum())
                rest = df_ext[df_ext["type"] != "talking"]
                learner_sec = float(rest[rest["norm_desc"].isin(LEARNER_CONTENT_SOURCES)]["timeSeconds"].sum())
                native_sec = float(rest[~rest["norm_desc"].isin(LEARNER_CONTENT_SOURCES)]["timeSeconds"].sum())
                return {"Content for Learners": learner_sec, "Native Content": native_sec, "Talking": talking_sec}

            for t_idx, (kind, value, _) in enumerate(tab_specs_ct):
                with tabs_ct[t_idx]:
                    if kind == "overall":
                        secs = compute_content_type_seconds(ext_all)
                        secs["Content for Learners"] += ds_overall_seconds
                    elif kind == "year":
                        year = int(value)
                        dfy = ext_all[ext_all["date"].dt.year == year]
                        secs = compute_content_type_seconds(dfy)
                        ds_year_seconds = float(
                            ds_monthly_seconds[pd.Index(ds_monthly_seconds.index).year == year].sum(),
                        )
                        secs["Content for Learners"] += ds_year_seconds
                    else:  # month
                        mper = value  # pd.Period
                        dfm = ext_all[ext_all["date"].dt.to_period("M") == mper]
                        secs = compute_content_type_seconds(dfm)
                        secs["Content for Learners"] += float(ds_monthly_seconds.get(mper, 0.0))

                    rows = [{"Category": k, "seconds": v} for k, v in secs.items() if v > 0]
                    if not rows:
                        st.write("No data for this period.")
                    else:
                        dft = pd.DataFrame(rows).sort_values("seconds", ascending=False).reset_index(drop=True)
                        dft["Time"] = (dft["seconds"] / 3600).apply(format_time)
                        dft["#"] = range(1, len(dft) + 1)
                        st.dataframe(dft[["#", "Category", "Time"]].set_index("#"), use_container_width=True)

            # Separate talking first (it bypasses description classification)
            if not ext_all.empty:
                talking_by_day = (
                    ext_all[ext_all["type"] == "talking"]
                    .groupby("date")["timeSeconds"]
                    .sum()
                )
                rest = ext_all[ext_all["type"] != "talking"]
                learner_ext_by_day = (
                    rest[rest["norm_desc"].isin(LEARNER_CONTENT_SOURCES)]
                    .groupby("date")["timeSeconds"]
                    .sum()
                )
                native_ext_by_day = (
                    rest[~rest["norm_desc"].isin(LEARNER_CONTENT_SOURCES)]
                    .groupby("date")["timeSeconds"]
                    .sum()
                )
            else:
                talking_by_day = pd.Series(dtype="float64")
                learner_ext_by_day = pd.Series(dtype="float64")
                native_ext_by_day = pd.Series(dtype="float64")

            # DS is always considered learner content
            learner_total_by_day = (
                ds_series.add(learner_ext_by_day, fill_value=0.0)
                .reindex(full_days, fill_value=0.0)
            )
            native_total_by_day = native_ext_by_day.reindex(full_days, fill_value=0.0)
            talking_total_by_day = talking_by_day.reindex(full_days, fill_value=0.0)

            st.markdown("#### Daily Cumulative by Content Type")
            tab_cum, tab_avg = st.tabs(["Cumulative", "30-day Average"])

            with tab_cum:
                # Cumulative totals
                cum_df = pd.DataFrame(
                    {
                        "date": full_days,
                        "Content for Learners": learner_total_by_day.cumsum() / 3600.0,
                        "Native Content": native_total_by_day.cumsum() / 3600.0,
                        "Talking": talking_total_by_day.cumsum() / 3600.0,
                    },
                )
                cum_long = cum_df.melt(id_vars=["date"], var_name="Category", value_name="cum_hours")

                fig_content = px.line(
                    cum_long,
                    x="date",
                    y="cum_hours",
                    color="Category",
                    category_orders={"Category": ["Content for Learners", "Native Content", "Talking"]},
                    color_discrete_map={
                        "Content for Learners": COLOUR_PALETTE["primary"],
                        "Native Content": COLOUR_PALETTE["30day_avg"],
                        "Talking": COLOUR_PALETTE["7day_avg"],
                    },
                    labels={"date": "Date", "cum_hours": "Cumulative Hours"},
                    title=None,
                )
                fig_content.update_traces(line={"width": 3})
                fig_content.update_layout(
                    height=450,
                    xaxis_title="Date",
                    yaxis_title="Cumulative Hours",
                    legend_title="Category",
                    margin={"l": 10, "r": 10, "t": 10, "b": 0},
                )
                fig_content.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_content, use_container_width=True)

            with tab_avg:
                # Daily totals in hours, then 30-day rolling averages
                daily_hours = pd.DataFrame(
                    {
                        "date": full_days,
                        "Content for Learners": learner_total_by_day / 3600.0,
                        "Native Content": native_total_by_day / 3600.0,
                        "Talking": talking_total_by_day / 3600.0,
                    },
                ).set_index("date")

                ma30 = daily_hours.rolling(30, min_periods=1).mean().reset_index()
                ma30_long = ma30.melt(id_vars=["date"], var_name="Category", value_name="hours")

                fig_avg = px.line(
                    ma30_long,
                    x="date",
                    y="hours",
                    color="Category",
                    category_orders={"Category": ["Content for Learners", "Native Content", "Talking"]},
                    color_discrete_map={
                        "Content for Learners": COLOUR_PALETTE["primary"],
                        "Native Content": COLOUR_PALETTE["30day_avg"],
                        "Talking": COLOUR_PALETTE["7day_avg"],
                    },
                    labels={"date": "Date", "hours": "Avg Hours/Day"},
                    title=None,
                )
                fig_avg.update_traces(line={"width": 3})
                fig_avg.update_layout(
                    height=450,
                    xaxis_title="Date",
                    yaxis_title="Avg Hours/Day",
                    legend_title="Category",
                    margin={"l": 10, "r": 10, "t": 10, "b": 0},
                )
                fig_avg.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_avg, use_container_width=True)

            # ============================
            # Spanish Variant Leaderboard
            # ============================
            st.markdown("#### Spanish Variant Leaderboard")
            tab_specs_var: list[tuple[str, object, str]] = [("overall", None, "Overall")]
            if years:
                tab_specs_var += [("year", y, str(y)) for y in years]
            tab_specs_var += [("month", m, m.strftime("%b %Y")) for m in months]
            tabs_var = st.tabs([label for _, _, label in tab_specs_var])

            LATIN_AMERICA_100_SOURCES = {
                "duolingo podcast", "babbel", "y tu mamá tambien", "universo curioso de la nasa",
                "preply con jose", "charlas hispanas", "un mundo inmenso", "hola spanish",
                "spanish boost", "estetica unisex", "luisito comunica"
            }
            LATIN_AMERICA_50_SOURCES = {"easy spanish", "<old podcasts>", "<old watching>"}

            def compute_variant_seconds(df_ext: pd.DataFrame) -> dict[str, float]:
                if df_ext.empty:
                    return {"Latin America Spanish": 0.0, "Spanish of Spain": 0.0}
                la100 = df_ext[df_ext["norm_desc"].isin(LATIN_AMERICA_100_SOURCES)]["timeSeconds"].sum()
                la50 = df_ext[df_ext["norm_desc"].isin(LATIN_AMERICA_50_SOURCES)]["timeSeconds"].sum()
                other = df_ext[
                    (~df_ext["norm_desc"].isin(LATIN_AMERICA_100_SOURCES))
                    & (~df_ext["norm_desc"].isin(LATIN_AMERICA_50_SOURCES))
                    & df_ext["norm_desc"].ne("")
                ]["timeSeconds"].sum()
                la_ext = float(la100) + float(la50) * 0.5
                sp_ext = float(other) + float(la50) * 0.5
                return {"Latin America Spanish": la_ext, "Spanish of Spain": sp_ext}

            for t_idx, (kind, value, _) in enumerate(tab_specs_var):
                with tabs_var[t_idx]:
                    if kind == "overall":
                        secs = compute_variant_seconds(ext_all)
                        secs["Latin America Spanish"] += ds_overall_seconds * 0.60
                        secs["Spanish of Spain"] += ds_overall_seconds * 0.40
                    elif kind == "year":
                        year = int(value)
                        dfy = ext_all[ext_all["date"].dt.year == year]
                        secs = compute_variant_seconds(dfy)
                        ds_year_seconds = float(ds_monthly_seconds[pd.Index(ds_monthly_seconds.index).year == year].sum())
                        secs["Latin America Spanish"] += ds_year_seconds * 0.60
                        secs["Spanish of Spain"] += ds_year_seconds * 0.40
                    else:
                        mper = value  # pd.Period
                        dfm = ext_all[ext_all["date"].dt.to_period("M") == mper]
                        secs = compute_variant_seconds(dfm)
                        ds_m = float(ds_monthly_seconds.get(mper, 0.0))
                        secs["Latin America Spanish"] += ds_m * 0.60
                        secs["Spanish of Spain"] += ds_m * 0.40

                    rows = [{"Category": k, "seconds": v} for k, v in secs.items() if v > 0]
                    if not rows:
                        st.write("No data for this period.")
                    else:
                        dft = pd.DataFrame(rows).sort_values("seconds", ascending=False).reset_index(drop=True)
                        dft["Time"] = (dft["seconds"] / 3600).apply(format_time)
                        dft["#"] = range(1, len(dft) + 1)
                        st.dataframe(dft[["#", "Category", "Time"]].set_index("#"), use_container_width=True)


            # Daily cumulative by Spanish variant (Latin America vs Spain)
            st.markdown("#### Daily Cumulative by Spanish Variant")
            # Variant source sets (same as count_external_hours.py)
            LATIN_AMERICA_100_SOURCES = {
                "duolingo podcast", "babbel", "y tu mamá tambien", "universo curioso de la nasa",
                "preply con jose", "charlas hispanas", "un mundo inmenso", "hola spanish",
                "spanish boost", "estetica unisex",
            }
            LATIN_AMERICA_50_SOURCES = {"easy spanish", "<old podcasts>", "<old watching>"}

            if not ext_all.empty:
                la100 = ext_all[ext_all["norm_desc"].isin(LATIN_AMERICA_100_SOURCES)]
                la50 = ext_all[ext_all["norm_desc"].isin(LATIN_AMERICA_50_SOURCES)]
                other = ext_all[
                    (~ext_all["norm_desc"].isin(LATIN_AMERICA_100_SOURCES))
                    & (~ext_all["norm_desc"].isin(LATIN_AMERICA_50_SOURCES))
                    & ext_all["norm_desc"].ne("")
                ]
                la100_by_day = la100.groupby("date")["timeSeconds"].sum()
                la50_by_day = la50.groupby("date")["timeSeconds"].sum()
                spain_by_day = other.groupby("date")["timeSeconds"].sum()
                la_ext_by_day = la100_by_day.add(la50_by_day.mul(0.5), fill_value=0.0)
                spain_ext_by_day = spain_by_day.add(la50_by_day.mul(0.5), fill_value=0.0)
            else:
                la_ext_by_day = pd.Series(dtype="float64")
                spain_ext_by_day = pd.Series(dtype="float64")

            # Dreaming Spanish split: 60% Latin America, 40% Spain (per count_external_hours.py)
            la_total_by_day = (
                ds_series.mul(0.60).add(la_ext_by_day, fill_value=0.0).reindex(full_days, fill_value=0.0)
            )
            spain_total_by_day = (
                ds_series.mul(0.40).add(spain_ext_by_day, fill_value=0.0).reindex(full_days, fill_value=0.0)
            )

            # Tabs: Cumulative vs 30-day Average
            tab_var_cum, tab_var_avg = st.tabs(["Cumulative", "30-day Average"])

            with tab_var_cum:
                # Cumulative hours
                cum_variant_df = pd.DataFrame(
                    {
                        "date": full_days,
                        "Latin America Spanish": la_total_by_day.cumsum() / 3600.0,
                        "Spanish of Spain": spain_total_by_day.cumsum() / 3600.0,
                    },
                )
                cum_variant_long = cum_variant_df.melt(
                    id_vars=["date"],
                    var_name="Variant",
                    value_name="cum_hours",
                )

                fig_variant = px.line(
                    cum_variant_long,
                    x="date",
                    y="cum_hours",
                    color="Variant",
                    category_orders={"Variant": ["Latin America Spanish", "Spanish of Spain"]},
                    color_discrete_map={
                        "Latin America Spanish": COLOUR_PALETTE["7day_avg"],
                        "Spanish of Spain": COLOUR_PALETTE["30day_avg"],
                    },
                    labels={"date": "Date", "cum_hours": "Cumulative Hours"},
                    title=None,
                )
                fig_variant.update_traces(line={"width": 3})
                fig_variant.update_layout(
                    height=450,
                    xaxis_title="Date",
                    yaxis_title="Cumulative Hours",
                    legend_title="Variant",
                    margin={"l": 10, "r": 10, "t": 10, "b": 0},
                )
                fig_variant.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_variant, use_container_width=True)

            with tab_var_avg:
                # 30-day rolling average hours/day by variant
                daily_variant_hours = pd.DataFrame(
                    {
                        "date": full_days,
                        "Latin America Spanish": la_total_by_day / 3600.0,
                        "Spanish of Spain": spain_total_by_day / 3600.0,
                    },
                ).set_index("date")
                ma30_var = daily_variant_hours.rolling(30, min_periods=1).mean().reset_index()
                ma30_var_long = ma30_var.melt(id_vars=["date"], var_name="Variant", value_name="hours")

                fig_variant_avg = px.line(
                    ma30_var_long,
                    x="date",
                    y="hours",
                    color="Variant",
                    category_orders={"Variant": ["Latin America Spanish", "Spanish of Spain"]},
                    color_discrete_map={
                        "Latin America Spanish": COLOUR_PALETTE["7day_avg"],
                        "Spanish of Spain": COLOUR_PALETTE["30day_avg"],
                    },
                    labels={"date": "Date", "hours": "Avg Hours/Day"},
                    title=None,
                )
                fig_variant_avg.update_traces(line={"width": 3})
                fig_variant_avg.update_layout(
                    height=450,
                    xaxis_title="Date",
                    yaxis_title="Avg Hours/Day",
                    legend_title="Variant",
                    margin={"l": 10, "r": 10, "t": 10, "b": 0},
                )
                fig_variant_avg.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_variant_avg, use_container_width=True)

            # ==========================================
            # Activity Type Leaderboard (Watch/Listen/Talk)
            # ==========================================
            st.markdown("#### Activity Type Leaderboard")
            tab_specs_act: list[tuple[str, object, str]] = [("overall", None, "Overall")]
            if years:
                tab_specs_act += [("year", y, str(y)) for y in years]
            tab_specs_act += [("month", m, m.strftime("%b %Y")) for m in months]
            tabs_act = st.tabs([label for _, _, label in tab_specs_act])

            cutoff_date = pd.Timestamp(2025, 4, 30).date()

            def ds_listen_watch_for_month(period_m: pd.Period) -> tuple[float, float]:
                ds_m = float(ds_monthly_seconds.get(period_m, 0.0))
                rep_date = period_m.to_timestamp().date()
                if rep_date > cutoff_date:
                    return ds_m * 0.80, ds_m * 0.20
                return ds_m * 0.50, ds_m * 0.50

            def ds_listen_watch_over_year(y: int) -> tuple[float, float]:
                listen, watch = 0.0, 0.0
                for mper, val in ds_monthly_seconds.items():
                    if mper.year == y:
                        l, w = ds_listen_watch_for_month(mper)
                        listen += l
                        watch += w
                return listen, watch

            def ds_listen_watch_overall() -> tuple[float, float]:
                listen, watch = 0.0, 0.0
                for mper, _ in ds_monthly_seconds.items():
                    l, w = ds_listen_watch_for_month(mper)
                    listen += l
                    watch += w
                return listen, watch

            def compute_activity_seconds(df_ext: pd.DataFrame) -> dict[str, float]:
                if df_ext.empty:
                    return {"Watching": 0.0, "Listening": 0.0, "Talking": 0.0}
                listen_ext = float(df_ext[df_ext["type"] == "listening"]["timeSeconds"].sum())
                watch_ext = float(df_ext[df_ext["type"] == "watching"]["timeSeconds"].sum())
                talk_ext = float(df_ext[df_ext["type"] == "talking"]["timeSeconds"].sum())
                return {"Watching": watch_ext, "Listening": listen_ext, "Talking": talk_ext}

            for t_idx, (kind, value, _) in enumerate(tab_specs_act):
                with tabs_act[t_idx]:
                    if kind == "overall":
                        secs = compute_activity_seconds(ext_all)
                        ds_l, ds_w = ds_listen_watch_overall()
                        secs["Listening"] += ds_l
                        secs["Watching"] += ds_w
                    elif kind == "year":
                        year = int(value)
                        dfy = ext_all[ext_all["date"].dt.year == year]
                        secs = compute_activity_seconds(dfy)
                        ds_l, ds_w = ds_listen_watch_over_year(year)
                        secs["Listening"] += ds_l
                        secs["Watching"] += ds_w
                    else:
                        mper = value  # pd.Period
                        dfm = ext_all[ext_all["date"].dt.to_period("M") == mper]
                        secs = compute_activity_seconds(dfm)
                        ds_l, ds_w = ds_listen_watch_for_month(mper)
                        secs["Listening"] += ds_l
                        secs["Watching"] += ds_w

                    rows = [{"Category": k, "seconds": v} for k, v in secs.items() if v > 0]
                    if not rows:
                        st.write("No data for this period.")
                    else:
                        dft = pd.DataFrame(rows).sort_values("seconds", ascending=False).reset_index(drop=True)
                        dft["Time"] = (dft["seconds"] / 3600).apply(format_time)
                        dft["#"] = range(1, len(dft) + 1)
                        st.dataframe(dft[["#", "Category", "Time"]].set_index("#"), use_container_width=True)

            # Daily cumulative by activity type (Watching, Listening, Talking)
            st.markdown("#### Daily Cumulative by Activity Type (Watching, Listening, Talking)")
            # External by type per day
            if not ext_all.empty:
                listen_ext_by_day = (
                    ext_all[ext_all["type"] == "listening"].groupby("date")["timeSeconds"].sum()
                )
                watch_ext_by_day = (
                    ext_all[ext_all["type"] == "watching"].groupby("date")["timeSeconds"].sum()
                )
                talk_ext_by_day = (
                    ext_all[ext_all["type"] == "talking"].groupby("date")["timeSeconds"].sum()
                )
            else:
                listen_ext_by_day = pd.Series(dtype="float64")
                watch_ext_by_day = pd.Series(dtype="float64")
                talk_ext_by_day = pd.Series(dtype="float64")

            # Split Dreaming Spanish daily seconds between listening/watching using date-based ratios
            cutoff = pd.Timestamp(2025, 4, 30)  # up to and including this date: 50/50; after: 80/20
            ds_daily = ds_series.reindex(full_days, fill_value=0.0)
            ratio_listen = pd.Series(0.5, index=full_days)
            ratio_listen[full_days > cutoff] = 0.80
            ratio_watch = 1.0 - ratio_listen
            ds_listen_daily = ds_daily * ratio_listen
            ds_watch_daily = ds_daily * ratio_watch

            # Combine DS with externals (talking is external only)
            listen_total_by_day = ds_listen_daily.add(listen_ext_by_day, fill_value=0.0).reindex(full_days, fill_value=0.0)
            watch_total_by_day = ds_watch_daily.add(watch_ext_by_day, fill_value=0.0).reindex(full_days, fill_value=0.0)
            talking_total_by_day = talk_ext_by_day.reindex(full_days, fill_value=0.0)

            # Tabs: Cumulative vs 30-day Average
            tab_act_cum, tab_act_avg = st.tabs(["Cumulative", "30-day Average"])

            with tab_act_cum:
                cum_type_df = pd.DataFrame(
                    {
                        "date": full_days,
                        "Watching": watch_total_by_day.cumsum() / 3600.0,
                        "Listening": listen_total_by_day.cumsum() / 3600.0,
                        "Talking": talking_total_by_day.cumsum() / 3600.0,
                    },
                )
                cum_type_long = cum_type_df.melt(id_vars=["date"], var_name="Type", value_name="cum_hours")

                fig_types = px.line(
                    cum_type_long,
                    x="date",
                    y="cum_hours",
                    color="Type",
                    category_orders={"Type": ["Watching", "Listening", "Talking"]},
                    color_discrete_map={
                        "Watching": COLOUR_PALETTE["primary"],
                        "Listening": COLOUR_PALETTE["7day_avg"],
                        "Talking": COLOUR_PALETTE["30day_avg"],
                    },
                    labels={"date": "Date", "cum_hours": "Cumulative Hours"},
                    title=None,
                )
                fig_types.update_traces(line={"width": 3})
                fig_types.update_layout(
                    height=450,
                    xaxis_title="Date",
                    yaxis_title="Cumulative Hours",
                    legend_title="Type",
                    margin={"l": 10, "r": 10, "t": 10, "b": 0},
                )
                fig_types.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_types, use_container_width=True)

            with tab_act_avg:
                # 30-day rolling average hours/day by activity type
                daily_type_hours = pd.DataFrame(
                    {
                        "date": full_days,
                        "Watching": watch_total_by_day / 3600.0,
                        "Listening": listen_total_by_day / 3600.0,
                        "Talking": talking_total_by_day / 3600.0,
                    },
                ).set_index("date")
                ma30_type = daily_type_hours.rolling(30, min_periods=1).mean().reset_index()
                ma30_type_long = ma30_type.melt(id_vars=["date"], var_name="Type", value_name="hours")

                fig_types_avg = px.line(
                    ma30_type_long,
                    x="date",
                    y="hours",
                    color="Type",
                    category_orders={"Type": ["Watching", "Listening", "Talking"]},
                    color_discrete_map={
                        "Watching": COLOUR_PALETTE["primary"],
                        "Listening": COLOUR_PALETTE["7day_avg"],
                        "Talking": COLOUR_PALETTE["30day_avg"],
                    },
                    labels={"date": "Date", "hours": "Avg Hours/Day"},
                    title=None,
                )
                fig_types_avg.update_traces(line={"width": 3})
                fig_types_avg.update_layout(
                    height=450,
                    xaxis_title="Date",
                    yaxis_title="Avg Hours/Day",
                    legend_title="Type",
                    margin={"l": 10, "r": 10, "t": 10, "b": 0},
                )
                fig_types_avg.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_types_avg, use_container_width=True)

with st.container(border=True):
    # Text predictions
    current_hours = df["cumulative_hours"].iloc[-1]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Expected Milestone Dates")

        header_cols = st.columns([2, 3, 3, 3])
        with header_cols[0]:
            st.write("**Milestone**")
        with header_cols[1]:
            st.write("**Overall avg**")
        with header_cols[2]:
            st.write("**7-day avg**")
        with header_cols[3]:
            st.write("**30-day avg**")

        for milestone in MILESTONES:
            if current_hours < milestone:
                days_to_milestone = (
                    (milestone - current_hours) * 3600
                ) / avg_seconds_per_day
                days_to_milestone_7day = (
                    ((milestone - current_hours) * 3600) / current_7day_avg
                    if current_7day_avg > 0
                    else float("inf")
                )
                days_to_milestone_30day = (
                    ((milestone - current_hours) * 3600) / current_30day_avg
                    if current_30day_avg > 0
                    else float("inf")
                )

                predicted_date = df["date"].iloc[-1] + timedelta(days=days_to_milestone)
                predicted_date_7day = df["date"].iloc[-1] + timedelta(
                    days=days_to_milestone_7day,
                )
                predicted_date_30day = df["date"].iloc[-1] + timedelta(
                    days=days_to_milestone_30day,
                )

                cols = st.columns([2, 3, 3, 3])
                with cols[0]:
                    st.write(f"🗓️ {milestone}h")
                with cols[1]:
                    st.write(
                        f"{predicted_date.strftime('%Y-%m-%d')} "
                        f"({days_to_milestone:.0f}d)",
                    )
                with cols[2]:
                    st.write(
                        f"{predicted_date_7day.strftime('%Y-%m-%d')} "
                        f"({days_to_milestone_7day:.0f}d)",
                    )
                with cols[3]:
                    st.write(
                        f"{predicted_date_30day.strftime('%Y-%m-%d')} "
                        f"({days_to_milestone_30day:.0f}d)",
                    )
            else:
                cols = st.columns([2, 9])
                with cols[0]:
                    st.write(f"🗓️ {milestone}h")
                with cols[1]:
                    st.write("✅ Already achieved!")

    with col2:
        st.subheader("Progress Overview")
        for milestone in MILESTONES:
            if current_hours < milestone:
                percentage = (current_hours / milestone) * 100
                st.write(f"Progress to {milestone} hours: {percentage:.1f}%")
                st.progress(percentage / 100)

general_insights, best_days = st.columns(2)
with general_insights:  # noqa: SIM117
    with st.container(border=True):
        st.subheader("Insights")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Best day stats
            best_day_idx = df["seconds"].idxmax()
            best_day = df.loc[best_day_idx]
            st.metric(
                "Best Day",
                f"{(best_day['seconds'] / 60):.0f} min",
                f"{best_day['date'].strftime('%a %b %d')}",
            )
            # Add consistency metric
            days_watched = (df["seconds"] > 0).sum()
            consistency = (days_watched / len(df)) * 100
            st.metric(
                "Consistency",
                f"{consistency:.1f}%",
                f"{days_watched} of {len(df)} days",
            )

        with col2:
            # Streak information
            st.metric("Current Streak", f"{current_streak} days")

            st.metric(
                "Goal Streak",
                f"{current_goal_streak} days",
                f"Best: {longest_goal_streak} days",
            )

        with col3:
            # Achievement metrics
            total_time = df["seconds"].sum()
            milestone_count = sum(
                m <= df["cumulative_hours"].iloc[-1] for m in MILESTONES
            )
            st.metric(
                "Total Time",
                f"{(total_time / 60):.0f} min",
                f"{milestone_count} milestones reached",
                delta_color="off",
            )

            goal_rate = (goals_reached / total_days) * 100
            st.metric(
                "Goal Achievement",
                f"{goals_reached} days",
                f"{goal_rate:.1f}% of days",
            )

with best_days:  # noqa: SIM117
    with st.container(border=True):
        st.subheader("Best Days")
        best_days = get_best_days(result, num_days=5)
        if not best_days:
            st.write("Not enough data to show top 5 days.")
        else:
            for day in best_days:
                hours, remainder = divmod(day["timeSeconds"], 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = (
                    f"{int(hours):02d} hours {int(minutes):02d} minutes "
                    f"{int(seconds):02d} seconds"
                )
                st.write(f"**{day['date']}**: {time_str}")

with st.container(border=True):
    st.subheader("Averaged Insights")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Streak information
        avg_streak = streak_lengths.mean() if not streak_lengths.empty else 0
        st.metric(
            "Average Streak",
            f"{avg_streak:.1f} days",
            f"Best: {longest_streak} days",
        )

    with col2:
        # Time comparisons
        last_7_total = df.tail(7)["seconds"].sum()
        previous_7_total = df.iloc[-14:-7]["seconds"].sum() if len(df) >= 14 else 0  # noqa: PLR2004
        week_change = last_7_total - previous_7_total
        st.metric(
            "Last 7 Days Total",
            f"{(last_7_total / 60):.0f} min",
            f"{(week_change / 60):+.0f} min vs previous week",
        )

    with col3:
        weekly_avg = df.tail(7)["seconds"].mean()
        st.metric(
            "7-Day Average",
            f"{(weekly_avg / 60):.1f} min/day",
            f"{((weekly_avg - avg_seconds_per_day) / 60):+.1f} vs overall",
        )


with st.container(border=True):
    st.subheader("Tools")
    result = st.session_state.data
    st.download_button(
        label="📥 Export Data to CSV",
        data=df.to_csv(index=False),
        file_name="dreaming_spanish_data.csv",
        mime="text/csv",
    )

st.caption(
    f"Data range: {df['date'].min().strftime('%Y-%m-%d')} to {
        df['date'].max().strftime('%Y-%m-%d')
    }",
)
