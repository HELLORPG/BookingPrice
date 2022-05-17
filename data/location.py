# 用来处理预订信息中的位置信息

import os

from data import BookingInfo
from data import InfoFile
import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D


def location_scatter_from_booking_info_file(input_path: str, save_path=None, cmap="Blues", title="Location and Price"):
    """
    使用BookingInfo File输入，从而进行位置的可视化
    :param input_path: input file path.
    :param save_path: output file path.
    :param cmap: Color Map.
    :param title: Title.
    :return: None.
    """

    data_file = InfoFile(input_path)

    booking_infos = data_file.csv_to_booking_info()

    location_scatter_from_booking_infos(
        infos=booking_infos,
        save_path=save_path,
        cmap=cmap,
        title=title
    )

    return


def location_scatter_from_booking_infos(infos: list[BookingInfo], save_path=None, cmap="Blues", title="Location and Price"):
    """
    输入一个BookingInfo类的列表，以及存储路径，展示并且保存一个location的散点图。
    :param infos: BookingInfo list.
    :param save_path: plot save path.
    :param cmap: Color Map.
    :param title: 表头。
    :return: None
    """
    latitudes = list()
    longitudes = list()
    prices = list()

    for info in infos:
        longitude, latitude = info.get_location()
        latitudes.append(latitude)
        longitudes.append(longitude)
        prices.append(info.get_price())

    location_scatter_from_xyps(
        xs=longitudes,
        ys=latitudes,
        ps=prices,
        save_path=save_path,
        title=title,
        cmap=cmap
    )

    return


def location_scatter_from_xyps(xs: list, ys: list, ps: list, save_path=None, cmap="Blues", title="Location and Price"):
    """
    输入位置信息列表和价格列表，从而生成散点图并且保存在指定位置。
    :param xs: 位置的x轴坐标，一般是经度
    :param ys: 位置的y轴坐标，一般是纬度
    :param ps: 价格区间，0～5。
    :param save_path: 保存路径，如果不传入则代表不保存。
    :param cmap: 散点图使用的Color Map。
    :param title: title.
    :return: None
    """

    plt.figure(dpi=600)

    plt.scatter(xs, ys, s=0.1, c=ps, cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if save_path is None:
        plt.show()
    else:
        save_dir, save_filename = os.path.split(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)

    plt.close()

    return


if __name__ == '__main__':
    location_scatter_from_booking_info_file(input_path="../dataset/train.csv", save_path="../outputs/location.png")
