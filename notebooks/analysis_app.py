import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path, encoding="ISO-8859-1")
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    return data


# Filter data based on the number of recent seasons
def filter_recent_seasons(data, num_seasons):
    unique_seasons = sorted(data["Season"].unique(), reverse=True)
    selected_seasons = unique_seasons[:num_seasons]
    return data[data["Season"].isin(selected_seasons)]


# Main Streamlit app
def main():
    st.title("EPL Data Analysis(1993-94 to 2021-22)")

    # Load data
    file_path = "../data/epl_results.csv"  # Update with the correct file path if needed
    data = load_data(file_path)

    # Select number of recent seasons to analyze
    num_seasons = st.slider(
        "Select number of recent seasons to analyze",
        min_value=1,
        max_value=len(data["Season"].unique()),
        value=3,
    )
    filtered_data = filter_recent_seasons(data, num_seasons)

    # Select analysis option
    analysis_option = st.selectbox(
        "Select Analysis Option",
        [
            "Team Performance",
            "Goal Distribution",
            "Home/Away Performance",
            "Seasonal Performance",
            "Team Averages Heatmap",
        ],
    )

    if analysis_option == "Team Performance":
        st.subheader("Team Performance Analysis")
        team = st.selectbox("Select Team", filtered_data["HomeTeam"].unique())
        team_data = filtered_data[
            (filtered_data["HomeTeam"] == team) | (filtered_data["AwayTeam"] == team)
        ]

        # Calculate performance
        wins = team_data[
            ((team_data["HomeTeam"] == team) & (team_data["FTR"] == "H"))
            | ((team_data["AwayTeam"] == team) & (team_data["FTR"] == "A"))
        ].shape[0]
        draws = team_data[team_data["FTR"] == "D"].shape[0]
        losses = team_data[
            ((team_data["HomeTeam"] == team) & (team_data["FTR"] == "A"))
            | ((team_data["AwayTeam"] == team) & (team_data["FTR"] == "H"))
        ].shape[0]

        st.write(f"Total Matches: {team_data.shape[0]}")
        st.write(f"Wins: {wins}")
        st.write(f"Draws: {draws}")
        st.write(f"Losses: {losses}")
        st.write(f"Winning Rate: {wins/team_data.shape[0]:.2f}")

        # Plot performance
        fig, ax = plt.subplots()
        ax.bar(
            ["Wins", "Draws", "Losses"],
            [wins, draws, losses],
            color=["green", "blue", "red"],
        )
        st.pyplot(fig)

    elif analysis_option == "Goal Distribution":
        st.subheader("Goal Distribution Analysis")
        team = st.selectbox("Select Team", filtered_data["HomeTeam"].unique())
        home_goals = filtered_data[filtered_data["HomeTeam"] == team]["FTHG"]
        away_goals = filtered_data[filtered_data["AwayTeam"] == team]["FTAG"]

        # Plot goal distribution
        fig, ax = plt.subplots()
        ax.hist(
            [home_goals, away_goals],
            bins=range(10),
            label=["Home Goals", "Away Goals"],
            color=["green", "red"],
            alpha=0.7,
            stacked=True,
        )
        ax.legend()
        ax.set_xlabel("Goals")
        ax.set_ylabel("Number of Matches")
        st.pyplot(fig)

    elif analysis_option == "Home/Away Performance":
        st.subheader("Home/Away Performance Comparison")
        team = st.selectbox("Select Team", filtered_data["HomeTeam"].unique())
        home_wins = filtered_data[
            (filtered_data["HomeTeam"] == team) & (filtered_data["FTR"] == "H")
        ].shape[0]
        away_wins = filtered_data[
            (filtered_data["AwayTeam"] == team) & (filtered_data["FTR"] == "A")
        ].shape[0]
        home_matches = filtered_data[filtered_data["HomeTeam"] == team].shape[0]
        away_matches = filtered_data[filtered_data["AwayTeam"] == team].shape[0]

        home_win_rate = home_wins / home_matches if home_matches > 0 else 0
        away_win_rate = away_wins / away_matches if away_matches > 0 else 0

        st.write(f"Home Win Rate: {home_win_rate:.2f}")
        st.write(f"Away Win Rate: {away_win_rate:.2f}")

        # Plot home/away performance
        fig, ax = plt.subplots()
        ax.bar(
            ["Home Wins", "Away Wins"],
            [home_win_rate, away_win_rate],
            color=["blue", "orange"],
        )
        st.pyplot(fig)

    elif analysis_option == "Seasonal Performance":
        st.subheader("Seasonal Performance Analysis")
        team = st.selectbox("Select Team", filtered_data["HomeTeam"].unique())
        team_data = filtered_data[
            (filtered_data["HomeTeam"] == team) | (filtered_data["AwayTeam"] == team)
        ]

        # Calculate seasonal performance
        season_performance = (
            team_data.groupby("Season")
            .apply(
                lambda x: pd.Series(
                    {
                        "Wins": ((x["HomeTeam"] == team) & (x["FTR"] == "H")).sum()
                        + ((x["AwayTeam"] == team) & (x["FTR"] == "A")).sum(),
                        "Draws": (x["FTR"] == "D").sum(),
                        "Losses": ((x["HomeTeam"] == team) & (x["FTR"] == "A")).sum()
                        + ((x["AwayTeam"] == team) & (x["FTR"] == "H")).sum(),
                    }
                )
            )
            .reset_index()
        )

        st.write(season_performance)

        # Plot seasonal performance
        fig, ax = plt.subplots()
        ax.plot(
            season_performance["Season"],
            season_performance["Wins"],
            label="Wins",
            color="green",
        )
        ax.plot(
            season_performance["Season"],
            season_performance["Draws"],
            label="Draws",
            color="blue",
        )
        ax.plot(
            season_performance["Season"],
            season_performance["Losses"],
            label="Losses",
            color="red",
        )
        ax.legend()
        ax.set_xlabel("Season")
        ax.set_ylabel("Number of Matches")
        st.pyplot(fig)

    elif analysis_option == "Team Averages Heatmap":
        st.subheader("Team Averages Heatmap")
        avg_data = (
            filtered_data.groupby("HomeTeam")
            .agg(
                {
                    "FTHG": "mean",
                    "FTAG": "mean",
                    "HS": "mean",
                    "AS": "mean",
                    "HST": "mean",
                    "AST": "mean",
                    "HC": "mean",
                    "AC": "mean",
                    "HF": "mean",
                    "AF": "mean",
                    "HY": "mean",
                    "AY": "mean",
                    "HR": "mean",
                    "AR": "mean",
                }
            )
            .reset_index()
        )

        avg_data = avg_data.set_index("HomeTeam")

        # st.write("Average statistics for selected seasons")
        # st.dataframe(avg_data)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(avg_data, annot=True, cmap="YlGnBu", ax=ax, fmt=".2f")
        # title
        ax.set_title("Average Statistics for Selected Seasons")
        st.pyplot(fig)


if __name__ == "__main__":
    main()
