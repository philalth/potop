"""
Module to generate csv files for specific areas.
"""
# pylint: skip-file
import dask.dataframe as dd

if __name__ == '__main__':
    dataframe = dd.read_csv('../../data/On-street_Car_Parking_Sensor_Data_-_2017.csv')
    dataframe = dataframe.loc[dataframe['Area'] == 'Queensberry']
    dataframe.to_csv('../../data/Queensberry.csv', single_file=True)
