# 用来处理预订信息中的位置信息

from data import BookingInfo
from data import DataFile
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    train_file = DataFile("../dataset/train.csv")
    booking_infos = train_file.csv_to_booking_info()

    location_x = list()
    location_y = list()
    price = list()
    for booking_info in booking_infos:
        location_x.append(booking_info.latitude)
        location_y.append(booking_info.longitude)
        price.append(booking_info.get_price())

    fig = plt.figure(dpi=400)
    # ax = Axes3D(fig)
    # for i in range(0, len(price)):
    #     plt.scatter(location_x[i], location_y[i], s=0.1, c=price[i], cmap="summer", label=price[i])
    #     print(i)
    # 上述的方法速度太慢了

    label_str = [str(p) for p in price]
    plt.scatter(location_x, location_y, s=0.1, c=price, cmap="Reds")

    # colmaps = plt.get_cmap('Reds')
    # print(type(colmaps))
    # for p in range(0, 6):
    #     this_price_infos = list()
    #     for i in range(0, len(price)):
    #         if price[i] == p:
    #             this_price_infos.append(i)
    #     this_price_xs = list()
    #     this_price_ys = list()
    #     cs = list()
    #     for info_index in this_price_infos:
    #         this_price_xs.append(location_x[info_index])
    #         this_price_ys.append(location_y[info_index])
    #         cs.append(p)
    #     plt.scatter(this_price_xs, this_price_ys, s=0.1, c=cs, label=p, cmap=colmaps[0.5])

    # print()
    plt.colorbar()
    plt.show()
