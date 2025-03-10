import pandas as pd
from model import ModelTrainer  # Import the class

# Load dataset
df = pd.read_csv("../data/engineered/engineered_data.csv")  # Adjust path if needed

# Initialize the model trainer
trainer = ModelTrainer(df)

# Train using Linear Regression
print("Training with Linear Regression...")
trainer.train_regression_model(model_type="linear")

# Train using Random Forest
print("Training with Random Forest...")
trainer.train_regression_model(model_type="random_forest")

# Save the best model
trainer.save_model("../saved_models/nba_stat_predictor.pkl")
