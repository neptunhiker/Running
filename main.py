import datetime
import pandas as pd

from objects import Analysis, DataImport, Database

if __name__ == '__main__':
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)

    db = Database(database_path="database//running.db")
    di = DataImport()
    di.add_runs_to_database(database=db)
    # db.export_runs_to_csv()

    ana = Analysis(database=db, weighting_factor=1, starting_date=datetime.date(2022, 12, 1), forecast=True)
    ana.show_runs()

    ana.plot_running_analytics()

    # todo: rolling distance should be calculated directly on the dataframe not in the plotting function