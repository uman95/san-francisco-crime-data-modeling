import datetime
import time

import numpy as np
import pandas as pd


def _get_dates(dates):
    yy = int(dates[:4])

    # print ("year = ",yy)

    # extracting month
    mm = int(dates[5:7])
    # print("month = ",mm)

    # extracting day
    dd = int(dates[8:10])
    # print("day = ",dd)

    # hour
    hh = int(dates[11:13])
    # print("hour = ",hh)

    # minutes
    minutes = int(dates[14:16])
    # print("minutes = ",minutes)

    seconds = int(dates[17:19])
    # print("seconds = ",seconds)

    return yy, mm, dd, hh, minutes, seconds


def _get_unix_time(time_as_tuple):
    timer = datetime.datetime(*time_as_tuple)
    timer = time.mktime(timer.timetuple())
    return timer


# Get the moment time the crime was happening
def _get_time_of_day(time_as_tuple):
    # moment = "Morning"
    hh = time_as_tuple[3]
    if hh >= 6 and hh < 13:
        moment = 0

    elif hh >= 13 and hh < 18:
        moment = 1

    else:
        moment = 2

    return moment


def _parse_data_to_onehot_encoding(datas_int, n_uniques):
    data_one_hot = np.eye(n_uniques)[datas_int]
    return data_one_hot

def _encode_date_time(date_time, time_of_day=True):
    n = int(len(date_time))
    offset = datetime.datetime(2003, 1, 1, 0, 0, 0) # start of data collection
    offset = time.mktime(offset.timetuple()) # to prevent possible numerical overflow
    if time_of_day:
        date_time_mat = np.zeros((n, 2), dtype=np.float64)
    else:
        date_time_mat = np.zeros((n, 1), dtype=np.float64)

    for i, entry in enumerate(date_time):
        tuple_time = _get_dates(entry)
        unix_time = _get_unix_time(tuple_time) - offset
        date_time_mat[i, 0] = unix_time
        if time_of_day:
            x = _get_time_of_day(tuple_time)
            date_time_mat[i, 1] = x
    return date_time_mat

def _one_hot_encode_strings(input_array):
    uniques = list(np.unique(input_array))
    uniques.sort()
    n = len(input_array)
    int_encoding = np.zeros((n), dtype=np.int64)
    n = len(input_array)
    for i in range(n):
        int_encoding[i] = uniques.index(input_array[i])
    return _parse_data_to_onehot_encoding(int_encoding, len(uniques))



def _design_matrix(pandas_frame, time_of_day=True, weekend_flag=True, normalized=True):
    n = int(len(pandas_frame))
    date = _encode_date_time(pandas_frame["Dates"], time_of_day)
    day_of_week = _one_hot_encode_strings(pandas_frame["DayOfWeek"])
    if weekend_flag:
        is_weekend = (day_of_week[:, 2] == 1) | (day_of_week[:, 3] == 1)
        day_of_week = np.hstack((day_of_week, is_weekend.reshape(-1, 1)))


    district = _one_hot_encode_strings(pandas_frame["PdDistrict"])
    geoloc = np.hstack((np.array(pandas_frame["X"]).reshape(-1,1),
                        np.array(pandas_frame["Y"]).reshape(-1,1)))
    crime = _one_hot_encode_strings(pandas_frame["Category"])

    if normalized:
        date[:, 0] = (date[:, 0] - date[:, 0])/date[:, 0].std()
    features = np.hstack((date, day_of_week, district, geoloc))

    return features, crime

def load_dataset(path_to_dataset, time_of_day=True, weekend_flag=True, normalized=True):
    """Return a tuple of two numpy arrays. Input features and target.

    Input features has N rows and the following columns:
    Unix_time (1 columns)
    [Time_of_day (3 columns onehot)]
    Day_of_week (7 columns onehot)
    [Is_weekend (1 column)]
    District (10 columns onehot)
    geoloc (2 columns (long, lat))

    The columns in square brackets will be abscent if their respective flags are set to False

    Target is a one hot encoding (dimension N x 39)

    ----------

    Example Usage:
    input_array, target = load_dataset("path/to/file/train.csv")"""
    data = pd.read_csv(path_to_dataset)
    return _design_matrix(data, time_of_day=time_of_day, weekend_flag=weekend_flag, normalized=normalized)


if __name__ == "__main__":
    d = load_dataset("../data/train.csv")
    np.set_printoptions(threshold=np.nan)
    print(d[0][1300:1400, 2:10])