import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor


# Load and preprocess the dataset
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path, encoding="ISO-8859-1")
    data = data[["Season", "DateTime", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data["Year"] = data["DateTime"].dt.year
    data["Month"] = data["DateTime"].dt.month
    data["HomeTeamGoals"] = data.groupby("HomeTeam")["FTHG"].transform(
        lambda x: x.rolling(10, 1).mean()
    )
    data["AwayTeamGoals"] = data.groupby("AwayTeam")["FTAG"].transform(
        lambda x: x.rolling(10, 1).mean()
    )
    return data


# Train Poisson regression models
@st.cache
def train_models(data):
    X = data[["HomeTeamGoals", "AwayTeamGoals"]]
    y_home = data["FTHG"]
    y_away = data["FTAG"]
    X_train, X_test, y_train_home, y_test_home, y_train_away, y_test_away = (
        train_test_split(X, y_home, y_away, test_size=0.2, random_state=42)
    )

    model_home = PoissonRegressor()
    model_home.fit(X_train, y_train_home)

    model_away = PoissonRegressor()
    model_away.fit(X_train, y_train_away)

    return model_home, model_away


# Predict goals
def predict_goals(home_team, away_team, data, model_home, model_away):
    home_goals = data[data["HomeTeam"] == home_team]["HomeTeamGoals"].values[-1]
    away_goals = data[data["AwayTeam"] == away_team]["AwayTeamGoals"].values[-1]
    pred_home_goals = model_home.predict([[home_goals, away_goals]])[0]
    pred_away_goals = model_away.predict([[home_goals, away_goals]])[0]
    return pred_home_goals, pred_away_goals


# Main Streamlit app
def main():
    st.title("EPL Goals Prediction")

    # Load data
    file_path = "../data/epl_results.csv"  # Update with the correct file path if needed
    data = load_data(file_path)

    # Train models
    model_home, model_away = train_models(data)

    # Select teams
    home_team = st.selectbox("Select Home Team", data["HomeTeam"].unique())
    away_team = st.selectbox("Select Away Team", data["AwayTeam"].unique())

    # Predict button
    if st.button("Predict"):
        pred_home_goals, pred_away_goals = predict_goals(
            home_team, away_team, data, model_home, model_away
        )
        st.write(f"Predicted goals for {home_team}: {pred_home_goals:.2f}")
        st.write(f"Predicted goals for {away_team}: {pred_away_goals:.2f}")


if __name__ == "__main__":
    main()
