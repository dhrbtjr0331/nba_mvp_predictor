import pandas as pd


class FeatureEngineer:
    """Class representing all necessary functions to engineer the suitable features"""

    def __init__(self, df):
        """Takes raw data as input of the constructor"""
        self.df = df.copy()

    def get_rolling_average(self, window_size=5):
        """Computes rolling average of a selected window size for each player."""
        df = self.df

        df.columns = df.columns.str.strip()

        # Sort by Player and Date
        df["Data"] = pd.to_datetime(df["Data"])
        df = df.sort_values(by=["Player", "Data"])

        # Select numerical stat columns to compute rolling averages
        stat_columns = ["PTS", "AST", "TRB", "FG%", "STL", "BLK"]

        # Compute rolling averages for each player
        for col in stat_columns:
            df[f"{col}_rolling_avg"] = (
                df.groupby("Player")[col]
                .rolling(window=window_size, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        # Update class variable
        self.df = df
        return self

    def get_opponent_defensive_stats(self):
        """Computes opponent defensive stats (average stats allowed per game)."""
        df = self.df

        df.columns = df.columns.str.strip()
        df["Data"] = pd.to_datetime(df["Data"])

        # Select numerical stat columns
        stat_columns = ["PTS", "AST", "TRB", "STL", "BLK"]

        # Compute opponent stats allowed
        opponent_defensive_stats = df.groupby("Opp")[stat_columns].mean()

        # Compute FG% Allowed (using FGM and FGA, not FG% mean)
        fg_stats = df.groupby("Opp")[["FG", "FGA"]].sum()
        fg_stats["FG%_allowed"] = fg_stats["FG"] / fg_stats["FGA"]

        # Merge FG% into opponent stats
        opponent_defensive_stats = opponent_defensive_stats.merge(
            fg_stats[["FG%_allowed"]], on="Opp"
        )

        # Rename columns to indicate they represent stats allowed
        opponent_defensive_stats = opponent_defensive_stats.add_suffix("_allowed").reset_index()

        # Merge back into main DataFrame
        self.df = self.df.merge(opponent_defensive_stats, on="Opp", how="left")
        return self

    def get_engineered_data(self):
        """Returns the final modified dataset with all features engineered."""
        return self.df


file_path = "data/raw/cleaned_data.csv"
df = pd.read_csv(file_path)

data_prep = FeatureEngineer(df)
data_prep.get_rolling_average(window_size=5)
data_prep.get_opponent_defensive_stats()
engineered_df = data_prep.get_engineered_data()
engineered_df.to_csv('engineered_data.csv')


