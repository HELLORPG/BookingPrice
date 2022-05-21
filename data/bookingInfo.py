# 用来表示房屋预定信息

import math

from enum import Enum
from math import floor
# from data.bookingToken import get_neighbor_tokenizer


class BookingInfo:
    def __init__(self, info=None):
        """
        使用info字典来初始化实例
        :param info: booking info in dict.
        """
        if info is None:
            self.description = None
            self.neighbor = None
            self.latitude = None  # 纬度
            self.longitude = None  # 经度
            self.type = None
            self.accommodates = None
            self.bathrooms = None
            self.bedrooms = None
            self.amenities = None  # 设施
            self.reviews = None
            self.review_rating = None
            self.review_A = None
            self.review_B = None
            self.review_C = None
            self.review_D = None
            self.instant_bookable = None

            # 用来表示标签，只有训练数据会有
            self.__has_price = False
            self.__price = None
        else:
            self.description = info["description"]
            self.neighbor = info["neighbor"]
            self.latitude = info["latitude"]
            self.longitude = info["longitude"]
            self.type = info["type"]
            self.accommodates = info["accommodates"]
            # self.bathrooms = info["bathrooms"]
            self.bathrooms = Bathroom(bathrooms=info["bathrooms"])
            self.bedrooms = info["bedrooms"]
            self.amenities = info["amenities"]
            self.reviews = info["reviews"]
            self.review_rating = info["review_rating"]
            self.review_A = info["review_A"]
            self.review_B = info["review_B"]
            self.review_C = info["review_C"]
            self.review_D = info["review_D"]
            self.instant_bookable = info["instant_bookable"]

            if "price" in info:
                self.__has_price = True
                self.__price = info["price"]
            else:
                self.__has_price = False

    def __str__(self):
        return self.to_dict().__str__()

    def to_dict(self):
        info = dict()
        info["description"] = self.description
        info["neighbor"] = self.neighbor
        info["latitude"] = self.latitude
        info["longitude"] = self.longitude
        info["type"] = self.type
        info["accommodates"] = self.accommodates
        info["bathrooms"] = self.bathrooms.to_str()
        info["bedrooms"] = self.bedrooms
        info["amenities"] = self.amenities
        info["reviews"] = self.reviews
        info["review_rating"] = self.review_rating
        info["review_A"] = self.review_A
        info["review_B"] = self.review_B
        info["review_C"] = self.review_C
        info["review_D"] = self.review_D
        info["instant_bookable"] = self.instant_bookable

        if self.__has_price:
            info["price"] = self.__price

        return info

    def get_price(self):
        if self.__has_price:
            return self.__price
        else:
            print("This info has no self.price.")
            exit(-1)

    def has_price(self):
        return self.__has_price

    def get_location(self):
        """
        返回经纬度位置信息
        :return: [longitude, latitude]
        """
        return self.longitude, self.latitude


class BathroomType(Enum):
    UNKNOWN = 0
    SHARED = 1
    PRIVATE = 2
    MISSING = 3


class Bathroom:
    """
    用来描述卫生间信息的类别
    """
    def __init__(self, bathrooms: str):
        """
        初始化一个关于卫生间信息的描述类。
        :param bathrooms: bathrooms information in string, always from class bookingInfo
        """
        if isinstance(bathrooms, float) or isinstance(bathrooms, int):
            if math.isnan(bathrooms):
                self.type = BathroomType.MISSING
                self.num = 0
                return
            else:
                self.type = BathroomType.UNKNOWN
                self.num = float(bathrooms)
                return

        if bathrooms[-1] == "s":    # 去除单复数的影响
            bathrooms = bathrooms[:-1]

        # print(bathrooms)
        assert bathrooms[-4:] == "bath"
        bathrooms = bathrooms[:-5]  # 截断最后的" bath"五个字符

        if len(bathrooms) > 6 and bathrooms[-6:] == "shared":
            self.type = BathroomType.SHARED
            bathrooms = bathrooms[:-7]
        elif len(bathrooms) > 7 and bathrooms[-7:] == "private":
            self.type = BathroomType.PRIVATE
            bathrooms = bathrooms[:-8]
        elif bathrooms == "Shared half":
            self.type = BathroomType.SHARED
            self.num = 0.5
            return
        elif bathrooms == "Private half":
            self.type = BathroomType.PRIVATE
            self.num = 0.5
            return
        else:
            self.type = BathroomType.UNKNOWN

        # if (type(eval(bathrooms)) == float) or (type(eval(bathrooms)) == int):
        #     self.num = float(bathrooms)
        # elif bathrooms == "Half":
        #     self.num = 0.5
        # else:
        #     print("Not support bathroom num: %s" % bathrooms)
        #     exit(-1)
        if bathrooms.isalpha():
            if bathrooms == "Half":
                self.num = 0.5
            else:
                print("Not support bathroom num: %s" % bathrooms)
                exit(-1)
        else:
            self.num = float(bathrooms)

    def to_str(self):
        """
        转化成为str类型。
        :return: bathrooms info in type string.
        """
        num_floor = floor(self.num)
        res = str(num_floor)
        if self.num - num_floor > 0.01:     # 判断是否是0.5结尾
            res += ".5"
        res += " "

        if self.type is BathroomType.SHARED:
            res += "shared bath"
        elif self.type is BathroomType.PRIVATE:
            res += "private bath"
        else:
            res += "bath"

        if num_floor > 1 or (num_floor == 1 and self.num - num_floor > 0.01):
            res += "s"

        return res


if __name__ == '__main__':
    # bathroom = Bathroom("0.5 private bath")
    # print(bathroom.type, bathroom.num)
    # print(bathroom.to_str())
    from data import InfoFile
    from data.bookingToken import get_neighbor_tokenizer
    all_infos = InfoFile("../dataset/split/val.csv").csv_to_booking_info()
    neighbor_token2str, neighbor_str2token = get_neighbor_tokenizer(all_infos)
    test_infos = InfoFile("../dataset/test.csv").csv_to_booking_info()
    print(get_neighbor_tokenizer(test_infos))
    print(neighbor_str2token)
    # print(BathroomType.MISSING.value)

