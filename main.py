import pandas as pd

from objects import Analysis, DataImport, Database

if __name__ == '__main__':
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)

    db = Database(database_path="database//running.db")
    di = DataImport()
    # pprint(di.data)
    # di.plot_rolling_distance()
    di.add_runs_to_database(database=db)
    # db.export_runs_to_csv()

    ana = Analysis(database=db, weighting_factor=1)
    ana.show_runs()

    ana.plot_rolling_load_for_runs()
    # ana.plot_scatter_loads()
    # ana.plot_rolling_distance()
