"""Define analysis data structures.

This module defines the data structures used to represent analysis results
from the Dreaming Spanish viewing data, including metrics like goals reached,
streaks, and the processed DataFrame containing all viewing statistics.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class AnalysisResult:
    """Represent the result of data analysis for the Dreaming Spanish application.

    Attributes:
        df (pd.DataFrame): A DataFrame containing the processed data.
        goals_reached (int): The total number of goals reached.
        total_days (int): The total number of days in the dataset.
        current_goal_streak (int): The current streak of consecutive days with goals
                                   reached.
        longest_goal_streak (int): The longest streak of consecutive days with goals
                                   reached.

    """

    df: pd.DataFrame
    goals_reached: int
    total_days: int
    current_goal_streak: int
    longest_goal_streak: int
