# 对数据文件进行读取和写入等操作

import pandas as pd


class DataFile:
    def __init__(self):
        """
        创建一个空的数据文件类
        """
        self.path = None

    def __init__(self, path: str):
        """
        传入文件路径的数据文件类
        :param path: path of data file
        """
        self.path = path

    def csv_to_list(self):
        """
        将csv文件中的每一条记录读取成为list形式
        :return: a line of data is a list.
        """
        assert len(self.path) >= 4
        assert self.path[-4:] == ".csv"
        data = pd.read_csv(self.path, sep=",", header=0)    # data in DataFrame
        data = data.values.tolist()
        return data

    def csv_to_info_dict(self):
        """
        将csv文件中的每一条记录读取成为dict形式
        :return: a line of data is an info dict (like class BookingPrice).
        """
        data = self.csv_to_list()

        assert len(data) >= 1
        assert len(data[0]) == 16 or len(data[0]) == 17     # test or train

        info_list = list()

        for line in data:
            info = dict()
            info["description"] = line[0]
            info["neighbor"] = line[1]
            info["latitude"] = line[2]
            info["longitude"] = line[3]
            info["type"] = line[4]
            info["accommodates"] = line[5]
            info["bathrooms"] = line[6]
            info["bedrooms"] = line[7]
            info["amenities"] = line[8]
            info["reviews"] = line[9]
            info["review_rating"] = line[10]
            info["review_A"] = line[11]
            info["review_B"] = line[12]
            info["review_C"] = line[13]
            info["review_D"] = line[14]
            info["instant_bookable"] = line[15]

            if len(line) == 17:
                info["price"] = line[16]

            info_list.append(info)

        return info_list


if __name__ == '__main__':
    train_file = DataFile("../dataset/train.csv")
    train_data = train_file.csv_to_info_dict()
    print(train_data)
    print(type(train_data))
    print(len(train_data))