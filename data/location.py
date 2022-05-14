# 用来处理预订信息中的位置信息

from data import BookingInfo
from data import DataFile


if __name__ == '__main__':
    train_file = DataFile("../dataset/train.csv")
    booking_infos = train_file.csv_to_booking_info()

    locations = list()
    for booking_info in booking_infos:
        location = [booking_info.latitude, booking_info.longitude]
        locations.append(location)

    print(locations)
