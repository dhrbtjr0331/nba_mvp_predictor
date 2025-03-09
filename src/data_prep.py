import pandas as pd


class DataPreparator:
    """Class representing all the data prep methods."""

    def __init__(self, df):
        self.df = df.copy()

    def handle_missing_values(self):
        """Fills missing values using previous game data for the same player."""
        for col in self.df.columns:
            # Forward fill within each player's group
            self.df[col] = self.df.groupby("Player")[col].ffill()
        return self

    def get_cleaned_data(self):
        """Returns the cleaned DataFrame."""
        return self.df

file_path = "data/data.csv"
df = pd.read_csv(file_path)

data_prep = DataPreparator(df)

data_prep.handle_missing_values()
cleaned_df = data_prep.get_cleaned_data()

cleaned_df.to_csv('cleaned_data.csv')