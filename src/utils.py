"""Provide utility functions for managing and analyzing viewing progress data.

Includes methods for data processing, statistical insights, and forecasting future
learning milestones.
"""

from datetime import timedelta

import httpx
import pandas as pd
import streamlit as st

from src.model import AnalysisResult


def fetch_ds_data(token: str) -> dict | None:
    """Fetch data from the Dreaming Spanish API using the provided bearer token.

    Args:
        token (str): The bearer token for API authentication.

    Returns:
        dict or None: A dictionary containing the fetched data if successful,
                      otherwise None.

    """
    url = "https://www.dreamingspanish.com/.netlify/functions/dayWatchedTime"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = httpx.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:  # noqa: BLE001
        st.error(f"Error fetching data: {e!s}")
        return None


def get_initial_time(token: str) -> int | None:
    """Fetch the initial time watched from the Dreaming Spanish API.

    This function uses the API call to fetch the "external times" using a bearer token.
    This data includes the time watched before the user started Dreaming Spanish,
    which is acquired in the users onboarding.

    Args:
        token (str): The bearer token for API authentication.

    Returns:
        int or None: The initial time watched (in seconds) if successful,
                     otherwise None.

    """
    url = "https://www.dreamingspanish.com/.netlify/functions/externalTime"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = httpx.get(url, headers=headers)
        response.raise_for_status()

        return response.json()["externalTimes"][0]["timeSeconds"]
    except Exception as e:  # noqa: BLE001
        st.error(f"Error fetching initial time: {e!s}")
        return None


def fetch_external_times_df(token: str) -> pd.DataFrame | None:
    """Fetch and normalize external learning time entries."""
    url = "https://www.dreamingspanish.com/.netlify/functions/externalTime"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = httpx.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "externalTimes" not in payload:
            return None
        df = pd.DataFrame(payload["externalTimes"])
        if df.empty:
            return df
        # Ensure expected columns and dtypes
        df["date"] = pd.to_datetime(df["date"])
        if "timeSeconds" in df.columns:
            df["timeSeconds"] = pd.to_numeric(df["timeSeconds"], errors="coerce").fillna(0.0)
        if "description" not in df.columns:
            df["description"] = ""
        if "type" not in df.columns:
            df["type"] = ""
        return df.sort_values("date")
    except Exception as e:  # noqa: BLE001
        st.error(f"Error fetching external times: {e!s}")
        return None


def normalize_description(desc: str) -> str:
    """Normalize free-text description to group similar sources."""
    d = (desc or "").lower().strip()
    if not d:
        return d
    if "duo" in d:
        return "duolingo podcast"
    if "david" in d:
        return "david"
    if d == "slc":
        return "spanish language coach"
    return d


def format_time(total_hours: float) -> str:
    """Format a float of hours into 'X hours and Y minutes' (no seconds)."""
    if not pd.notna(total_hours) or total_hours <= 0:
        return "0 minutes"
    hours_part = int(total_hours)
    minutes_part = round((total_hours - hours_part) * 60)
    parts = []
    if hours_part > 0:
        parts.append(f"{hours_part} hour{'s' if hours_part != 1 else ''}")
    if minutes_part > 0:
        parts.append(f"{minutes_part} minute{'s' if minutes_part != 1 else ''}")
    return " and ".join(parts) if parts else "0 minutes"


def load_data(token: str) -> AnalysisResult | None:
    """Load and processes data from the Dreaming Spanish API.

    This function fetches data using the provided bearer token,
    converts the data into a DataFrame, and processes it to include goal tracking
    metrics such as total days, goals reached, current goal streak, and longest
    goal streak.

    Args:
        token (str): The bearer token for API authentication.

    Returns:
        AnalysisResult or None: An AnalysisResult object containing the processed data
                                if successful, otherwise None.

    """
    if not token or not token.strip():
        return None

    api_data = fetch_ds_data(token)
    if not api_data or not isinstance(api_data, list) or len(api_data) == 0:
        return None

    # Convert API data to DataFrame
    df = pd.DataFrame(api_data)
    df["date"] = pd.to_datetime(df["date"])

    # Drop UserId (unnecessary for our case)
    df = df.drop(columns=["userId"])

    # Create a complete date range
    df = df.set_index("date").asfreq("D").reset_index()
    df = df.rename(columns={"index": "date"})

    # Fill missing values with explicit types
    df = df.astype({"timeSeconds": "float64", "goalReached": "boolean"}).fillna(
        {"timeSeconds": 0.0, "goalReached": False},
    )

    # Sort by date
    df = df.sort_values("date")

    # Add goal tracking metrics
    total_days = len(df)
    goals_reached = df["goalReached"].sum()

    # Calculate current goal streak
    df["goal_streak_group"] = (~df["goalReached"]).cumsum()
    df["current_goal_streak"] = df.groupby("goal_streak_group")["goalReached"].cumsum()
    current_goal_streak = (
        df["current_goal_streak"].iloc[-1] if df["goalReached"].iloc[-1] else 0
    )

    # Calculate longest goal streak
    goal_streak_lengths = df[df["goalReached"]].groupby("goal_streak_group").size()
    longest_goal_streak = (
        goal_streak_lengths.max() if not goal_streak_lengths.empty else 0
    )

    # Fetch external sources data
    external_df = fetch_external_times_df(token)

    return AnalysisResult(
        df=df,
        goals_reached=goals_reached,
        total_days=total_days,
        current_goal_streak=current_goal_streak,
        longest_goal_streak=longest_goal_streak,
        external_df=external_df,
    )


def generate_future_predictions(
    df: pd.DataFrame,
    avg_seconds_per_day: float,
    target_hours: float,
) -> pd.DataFrame:
    """Generate future predictions based on historical data.

    Generate future predictions based on historical data and on
    a given average seconds per day, stopping when the target hours are reached.

    Args:
        df (pd.DataFrame): The existing DataFrame containing historical data.
        avg_seconds_per_day (float): The average seconds watched per day.
        target_hours (float): The target number of hours to predict up to.

    Returns:
        pd.DataFrame: A DataFrame containing future predictions with dates, seconds,
                      cumulative seconds, cumulative minutes, and cumulative hours.

    """
    if len(df) == 0:
        return pd.DataFrame()

    if avg_seconds_per_day <= 0:
        avg_seconds_per_day = 1  # Prevent division by zero

    last_date = df["date"].iloc[-1]
    last_cumulative_seconds = df["cumulative_seconds"].iloc[-1]
    current_hours = last_cumulative_seconds / 3600

    # Calculate how many days needed to reach target
    hours_remaining = target_hours - current_hours
    days_needed = int((hours_remaining * 3600 / avg_seconds_per_day) + 1)

    # Generate enough days to reach target
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=days_needed,
        freq="D",
    )

    future_seconds = pd.Series([avg_seconds_per_day] * len(future_dates))
    future_df = pd.DataFrame({"date": future_dates, "seconds": future_seconds})

    # Calculate cumulative values
    future_df["cumulative_seconds"] = future_seconds.cumsum() + last_cumulative_seconds
    future_df["cumulative_minutes"] = future_df["cumulative_seconds"] / 60
    future_df["cumulative_hours"] = future_df["cumulative_minutes"] / 60

    # Only keep predictions up to slightly above target hours
    future_df = future_df[future_df["cumulative_hours"] <= target_hours * 1.05]

    # Create a single row DataFrame for the last historical point
    last_point = pd.DataFrame(
        {
            "date": [last_date],
            "seconds": [df["seconds"].iloc[-1]],
            "cumulative_seconds": [last_cumulative_seconds],
            "cumulative_minutes": [last_cumulative_seconds / 60],
            "cumulative_hours": [last_cumulative_seconds / 3600],
        },
    )

    # Combine last historical point with future predictions
    return pd.concat([last_point, future_df], ignore_index=True)


def get_best_days(analysis_result: AnalysisResult, num_days: int = 5) -> list[dict]:
    """Identify and return the top N days with the most time spent.

    Args:
        analysis_result (AnalysisResult): The analysis result object.
        num_days (int): The number of top days to retrieve.

    Returns:
        list[dict]: A list of dictionaries, each representing a best day
                    with 'date' and 'timeSeconds'.
                    Returns an empty list if not enough data.

    """
    if analysis_result.df.empty or len(analysis_result.df) < num_days:
        return []  # Indicate not enough data

    # Sort by timeSeconds in descending order and get the top N
    best_days_df = analysis_result.df.sort_values(
        by="timeSeconds",
        ascending=False,
    ).head(num_days)

    # Convert to a list of dictionaries for easier display
    best_days_list = []
    for _, row in best_days_df.iterrows():
        best_days_list.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "timeSeconds": int(row["timeSeconds"]),  # Ensure integer for display
            },
        )
    return best_days_list
