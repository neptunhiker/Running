from contextlib import closing
from dataclasses import dataclass, field
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import sqlite3
from typing import List, Optional

import helpers

@dataclass
class Run:
    """
    This class represents a running activity.
    It has the following attributes:
        date (datetime.date): The date of the run.
        time (datetime.time): The time it took to complete the run.
        distance (float): The distance of the run in miles.
        avg_hr (float): The average heart rate during the run.
        max_hr (float): The maximum heart rate during the run.
        calories (int): The number of calories burned during the run.
        avg_cadence (int): The average cadence during the run.
    """
    date: datetime.date
    time: datetime.time
    distance: float
    avg_hr: float
    max_hr: float
    calories: int
    avg_cadence: int
    garmin_load: Optional[int] = field(default=None)
    runalyze_trimp: Optional[int] = field(default=None)


class Database:
    """A class for establishing a connection to a data base"""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        self.conn = None

    def connect_to_database(self) -> None:
        """Establish the connection to the data base"""

        self.conn = sqlite3.connect(self.database_path)
        self.conn.row_factory = sqlite3.Row

    def add_run(self, run: Run) -> None:
        """Insert run data into the data base"""

        self.connect_to_database()

        with closing(self.conn.cursor()) as cursor:

            # Select the data from the table
            the_id = f"{run.date} - {run.distance}"
            cursor.execute("SELECT * FROM Runs WHERE ID = ?", (the_id,))

            # Fetch the results
            results = cursor.fetchall()

            # If there are no results, the data does not already exist in the table
            if len(results) == 0:
                sql = "INSERT INTO Runs (ID, Distance, Date, Time, Avg_HR, Max_HR, Calories, Avg_cadence, Garmin_load," \
                      "Runalyze_trimp) " \
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                cursor.execute(sql, (
                    the_id, run.distance, run.date, str(run.time), int(run.avg_hr), int(run.max_hr),
                    int(run.calories), int(run.avg_cadence), run.garmin_load, run.runalyze_trimp))
                self.conn.commit()
                print(f"{run} successfully added to the database.")
            else:
                # print(f"{run} could not be added to the database as it seems to already exist.")
                pass

    def export_runs_to_csv(self):
        """
        This function exports data from a table called 'Runs' in an SQLite3 database
        to a CSV file with the current date and time in the file name in the format
        'YYYY-MM-DD HH-MM database backup of runs.csv'
        """
        self.connect_to_database()

        # Read data from the 'Runs' table into a DataFrame
        df = pd.read_sql_query("SELECT * FROM Runs", self.conn)

        # Get current date and time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d %H-%M")

        # Write the DataFrame to a CSV file
        df.to_csv(f"{date_time} database backup of runs.csv", index=False)

        print(f"Data exported successfully to {date_time} database backup of runs.csv")

        # Close the database connection
        self.conn.close()

    def get_runs(self) -> List[Run]:
        """Return a list of all runs found in the database"""

        sql = "SELECT * FROM Runs"
        self.connect_to_database()
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()

        runs = []
        for the_row in results:
            runs.append(self.create_run(sqlite3_row=the_row))

        return runs

    @staticmethod
    def create_run(sqlite3_row: sqlite3.Row) -> Run:
        """Create a Run object from an sqlite3.Row"""

        return Run(date=sqlite3_row["Date"],
                   time=sqlite3_row["Time"],
                   distance=sqlite3_row["Distance"],
                   avg_hr=sqlite3_row["Avg_HR"],
                   max_hr=sqlite3_row["Max_HR"],
                   calories=sqlite3_row["Calories"],
                   avg_cadence=sqlite3_row["Avg_cadence"]
                   )


class DataImport:

    def __init__(self):
        self.data = pd.DataFrame()
        self.df_distances = pd.DataFrame()
        self._import_data_from_csv()
        self._convert_string_values_to_numeric_values()
        self._separate_date_and_time()
        self._filter_for_time_frame()
        self._filter_for_running_only()
        self._create_df_distance()

    def _import_data_from_csv(self) -> None:
        """
        Import running data exported from Garmin
        :return: None
        """
        self.data = pd.read_csv("data//running.csv", encoding="latin-1", sep=";", decimal=",")

    def _convert_string_values_to_numeric_values(self) -> None:
        """
        Convert values that have been imported as strings into floats
        :return: None
        """
        # self.data['Distance'] = pd.to_numeric(self.data['Distance'].str.replace(',', '.'), errors='coerce')
        # self.data['Aerobic TE'] = pd.to_numeric(self.data['Aerobic TE'].str.replace(',', '.'), errors='coerce')

    def _filter_for_running_only(self) -> None:
        """
        Filter the data for running activities only
        :return: None
        """
        self.data = self.data[self.data['Activity Type'].isin(['Running'])]

    def _separate_date_and_time(self) -> None:
        """
        Separate date and time into two columns in the data dataframe
        :return: None
        """

        # Convert the 'Date' column to a datetime object
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d.%m.%y %H:%M')

        # Extract the date and time into separate columns
        self.data['Date_only'] = self.data['Date'].dt.date
        self.data['Time_only'] = self.data['Date'].dt.time

        # Delete the original Date column
        del self.data["Date"]

    def _filter_for_time_frame(self, start_date: str = "2022-07-01") -> None:
        """
        Filter the dataframe to start after the given start_date
        :param start_date: the date after which the dataframe will be shown
        :return: None
        """
        # convert the date column to datetime format
        self.data['Date_only'] = pd.to_datetime(self.data['Date_only'])

        # filter the DataFrame to keep only the rows where the date is larger than the startdate
        self.data = self.data[self.data['Date_only'] >= start_date]

    def _create_df_distance(self) -> None:
        """
        Create separate dataframes for specific data such as distances, heart rates etc.
        :return: None
        """

        self.df_distances = self.data[['Date_only', 'Distance']]

        # Create a date range that covers the entire period of interest
        date_range = pd.date_range(start=self.df_distances['Date_only'].min(), end=datetime.date.today(),
                                   freq='D')

        # group by date
        self.df_distances = self.df_distances.groupby('Date_only').sum().reset_index()

        # Reindex the DataFrame to align its index with the date range
        self.df_distances = self.df_distances.set_index('Date_only').reindex(date_range, fill_value=0).reset_index()
        self.df_distances = self.df_distances.rename(columns={'index': 'Date_only'})

    def add_runs_to_database(self, database: Database) -> None:
        """
        Add runs from dataimport to database
        :param database: an sqlite3 database object
        :return: None
        """

        # iterate through data
        for index, row in self.data.iterrows():
            date = row["Date_only"].date()
            time = row["Time_only"]
            distance = row["Distance"]
            avg_hr = row["Avg HR"]
            max_hr = row["Max HR"]
            calories = int(float(row["Calories"]))
            cadence = row["Avg Run Cadence"]

            run = Run(date, time, distance, avg_hr, max_hr, calories, cadence)
            database.add_run(run)

    def plot_rolling_distance(self, n: int = 7) -> None:
        """
        Plot the rolling distance for running over the last n days
        :param n: rolling window length
        :return: None
        """
        n1 = 42
        rolling_distance = self.df_distances.copy()
        rolling_distance[f'rolling_distance({n})'] = rolling_distance['Distance'].rolling(window=n).mean()
        rolling_distance[f'rolling_distance({n1})'] = rolling_distance['Distance'].rolling(window=n1).mean()

        fig, (ax) = plt.subplots(figsize=(12, 6))
        ax.fill_between(rolling_distance["Date_only"], rolling_distance[f'rolling_distance({n1})'], color='darkred',
                        label=f'Rolling distance over {n1} days')
        ax.plot(rolling_distance["Date_only"], rolling_distance[f'rolling_distance({n})'],
                label=f'Rolling distance over {n} days', color='black')
        # ax.plot(rolling_distance["Date_only"], rolling_distance[f'rolling_distance({n1})'],
        #         label=f'Rolling distance over {n1} days')
        ax.set_ylabel('Average distance per day in km')
        ax.set_title('Rolling Distance over time')
        ax.legend()
        fig.tight_layout()
        plt.show()


@dataclass
class Analysis:
    database: Database
    weighting_factor: float = 0.9
    all_runs: pd.DataFrame = field(init=False)
    grouped_runs: pd.DataFrame = field(init=False)
    starting_date: datetime.date = datetime.date(2022, 7, 1)

    def __post_init__(self):
        self._get_all_runs()
        self._group_runs()
        self._sort_by_date()
        self._filter_for_starting_date()
        self._add_rolling_loads()
        self._add_atl(b=self.weighting_factor)
        self._add_ctl(b=self.weighting_factor)
        self._add_tsb()

    def _get_all_runs(self) -> None:
        """
        Get all running data from database
        :return: None
        """
        self.database.connect_to_database()
        self.all_runs = pd.read_sql_query("SELECT * FROM Runs", self.database.conn)
        self.all_runs['Date'] = pd.to_datetime(self.all_runs['Date'])
        self.all_runs['Runalyze_trimp'] = pd.to_numeric(self.all_runs['Runalyze_trimp'], errors='coerce')

    def _group_runs(self) -> None:
        """
        Create a df for runs for every day in the past by grouping them by date
        :return:
        """
        # Create a date range that covers the entire period of interest
        date_range = pd.date_range(start=self.all_runs['Date'].min(), end=datetime.date.today(),
                                   freq='D')

        # group by date
        self.grouped_runs = self.all_runs.groupby('Date').sum().reset_index()

        # keep only relevant columns
        self.grouped_runs = self.grouped_runs[["Date", "Distance", "Calories", "Garmin_load", "Runalyze_trimp"]]

        # Reindex the DataFrame to align its index with the date range
        self.grouped_runs = self.grouped_runs.set_index('Date').reindex(date_range, fill_value=0).reset_index()
        self.grouped_runs = self.grouped_runs.rename(columns={'index': 'Date'})

    def _sort_by_date(self) -> None:
        """
        Sort the running data by date
        :return: None
        """
        self.grouped_runs = self.grouped_runs.sort_values(by="Date", ascending=True)

    def _filter_for_starting_date(self) -> None:
        """
        Filter the running data such that only data after the starting date are kept
        :return: None
        """

        # convert the date column to datetime format
        self.grouped_runs['Date'] = pd.to_datetime(self.grouped_runs['Date']).dt.date

        # filter the DataFrame to keep only the rows where the date is larger than the startdate
        self.grouped_runs = self.grouped_runs[self.grouped_runs['Date'] >= self.starting_date]

    def _add_rolling_loads(self) -> None:
        """
        Add rolling loads for Garmin load and Runalyze TRIMP to the data. It will be the rolling loads for 7 days and
        for 6 weeks. To make both comparable the sum of loads is calculated for a 7 day period for both rolling
        windows.
        :return: None
        """
        n0 = 7
        n1 = 42

        self.grouped_runs[f'rolling_garmin_load ({n0})'] = self.grouped_runs['Garmin_load'].rolling(window=n0).sum()
        self.grouped_runs[f'rolling_garmin_load ({n1})'] = self.grouped_runs['Garmin_load'].rolling(window=n1).sum() / 6
        self.grouped_runs[f'rolling_runalyze_trimp ({n0})'] = self.grouped_runs['Runalyze_trimp'].rolling(
            window=n0).sum()
        self.grouped_runs[f'rolling_runalyze_trimp ({n1})'] = self.grouped_runs['Runalyze_trimp'].rolling(
            window=n1).sum() / 6

    def _add_atl(self, b: float) -> None:
        """
        Add the ATL (Acute Training Load) as an exponentially weighted metric to the data
        :param b: the weighting factor
        :return: None
        """
        n0 = 7

        self.grouped_runs['ATL Garmin'] = 0
        self.grouped_runs['ATL Runalyze'] = 0
        self.grouped_runs.reset_index(inplace=True, drop=True)
        for i in range(n0, len(self.grouped_runs)):
            garmin_loads = self.grouped_runs.iloc[i - n0 + 1:i + 1]['Garmin_load'].tolist()
            garmin_loads.reverse()
            runalyze_trimps = self.grouped_runs.iloc[i - n0 + 1:i + 1]['Runalyze_trimp'].tolist()
            runalyze_trimps.reverse()
            self.grouped_runs.at[i, 'ATL Garmin'] = helpers.weighted_sum(garmin_loads, b, n0)
            self.grouped_runs.at[i, 'ATL Runalyze'] = helpers.weighted_sum(runalyze_trimps, b, n0)

        # self.grouped_runs['ATL Garmin'] = self.grouped_runs["Garmin_load"].rolling(
        #     window=n0).apply(lambda x: (x * np.power(b, np.arange(len(x)))).sum(), raw=False)
        # self.grouped_runs['ATL Runalyze'] = self.grouped_runs["Runalyze_trimp"].rolling(
        #     window=n0).apply(lambda x: (x * np.power(b, np.arange(len(x)))).sum(), raw=False)

    def _add_ctl(self, b: float) -> None:
        """
        Add the CTL (Chronic Training Load) as an exponentially weighted metric to the data
        :param b: the weighting factor
        :return: None
        """
        n0 = 42

        self.grouped_runs['CTL Garmin'] = 0
        self.grouped_runs['CTL Runalyze'] = 0
        self.grouped_runs.reset_index(inplace=True, drop=True)
        for i in range(n0, len(self.grouped_runs)):
            garmin_loads = self.grouped_runs.iloc[i - n0 + 1:i + 1]['Garmin_load'].tolist()
            garmin_loads.reverse()
            runalyze_trimps = self.grouped_runs.iloc[i - n0 + 1:i + 1]['Runalyze_trimp'].tolist()
            runalyze_trimps.reverse()
            self.grouped_runs.at[i, 'CTL Garmin'] = helpers.weighted_sum(garmin_loads, b, n0) / 6
            self.grouped_runs.at[i, 'CTL Runalyze'] = helpers.weighted_sum(runalyze_trimps, b, n0) / 6

        # self.grouped_runs[f'CTL Garmin'] = self.grouped_runs["Garmin_load"].rolling(
        #     window=n1).apply(lambda x: (x * np.power(b, np.arange(len(x)))).sum() / 6, raw=False)
        # self.grouped_runs[f'CTL Runalyze'] = self.grouped_runs["Runalyze_trimp"].rolling(
        #     window=n1).apply(lambda x: (x * np.power(b, np.arange(len(x)))).sum() / 6, raw=False)

    def _add_tsb(self) -> None:
        """
        Add the TSB (Training Stress Balance) to the data as the difference between the ATL and CTL
        :return: None
        """

        self.grouped_runs["TSB Garmin"] = self.grouped_runs["CTL Garmin"] - self.grouped_runs["ATL Garmin"]
        self.grouped_runs["TSB Runalyze"] = self.grouped_runs["CTL Runalyze"] - self.grouped_runs["ATL Runalyze"]

    def show_runs(self) -> None:
        """
        Print the dataframe containing all runs
        :return: None
        """
        pprint(self.grouped_runs)

    def plot_rolling_load_for_runs(self) -> None:
        """
        Plot the rolling load for running over the last n days
        :return: None
        """

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 6))
        ax0.fill_between(self.grouped_runs["Date"], self.grouped_runs['CTL Garmin'], color='darkred',
                         label='CTL Garmin')
        ax0.plot(self.grouped_runs["Date"], self.grouped_runs['ATL Garmin'],
                 label='ATL Garmin', color='black')
        ax0.set_ylabel('Garmin load')
        ax0.set_ylim(0, 1200)
        ax0.legend()

        ax1.fill_between(self.grouped_runs["Date"], self.grouped_runs['CTL Runalyze'], color='darkred',
                         label='CTL Runalyze')
        ax1.plot(self.grouped_runs["Date"], self.grouped_runs['ATL Runalyze'],
                 label='ATL Runalyze', color='black')
        ax1.set_ylabel('Runalyze TRIMP')
        ax1.set_ylim(0, 1200)
        ax1.legend()

        ax2.plot(self.grouped_runs["Date"], self.grouped_runs[f'CTL Garmin'],
                 label=f'CTL Garmin', color='darkred')
        ax2.plot(self.grouped_runs["Date"], self.grouped_runs[f'CTL Runalyze'],
                 label=f'CTL Runalyze', color='black')
        ax2.set_ylabel('Load & TRIMP')
        ax2.set_ylim(0, 1200)
        ax2.legend()

        ax3.plot(self.grouped_runs["Date"], self.grouped_runs["TSB Garmin"],
                 label=f'TSB Garmin', color='darkred')
        ax3.plot(self.grouped_runs["Date"], self.grouped_runs["TSB Runalyze"],
                 label=f'TSB Runalyze', color='black')

        ax3.set_ylabel('Load & TRIMP')
        ax3.set_ylim(-800, 800)
        ax3.legend()

        fig.tight_layout()
        plt.show()

    def plot_rolling_distance(self) -> None:
        """
        Plot the rolling distance in kms over n days
        :param n: number of days
        :return: None
        """
        self._sort_by_date()
        fig, ax = plt.subplots(figsize=(12, 6))

        for n in [42, 60]:
            label_n = f"Rolling distance over {n} days"
            rolling_distance = self.grouped_runs["Distance"].rolling(window=n).mean()
            ax.plot(self.grouped_runs.Date, rolling_distance, label=label_n)

        plt.legend()
        fig.tight_layout()
        plt.show()

    def plot_scatter_loads(self) -> None:
        """
        Create a scatter plot of different load data from Garmin and Runalyze
        :return: None
        """
        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(self.grouped_runs.Garmin_load, self.grouped_runs.Runalyze_trimp, color="darkred", s=16)

        # Add axis labels
        plt.xlabel('Garmin Load')
        plt.ylabel('Runalyze TRIMP')

        # Show the
        fig.tight_layout()
        plt.show()
