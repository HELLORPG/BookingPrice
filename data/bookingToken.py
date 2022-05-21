# 将bookingInfo类别编码之后得到的特征
# 便于后续的处理

from data import BookingInfo
from data.utils import strs_to_tokens


class BookingToken:
    """
    将房屋预定信息 bookingToken 进行序列化
    """
    def __init__(self, info: BookingInfo):
        """
        初始化一个BookingToken实例。尚且是None的数据需要通过更复杂的处理或经过One Hot的处理过程等。
        :param info: BookingInfo instance.
        """
        self.description = None
        self.neighbor = None
        self.latitude = info.latitude       # 纬度
        self.longitude = info.longitude     # 经度
        self.type = None
        self.accommodates = info.accommodates
        self.bathrooms = None
        self.bedrooms = info.bedrooms
        self.amenities = None  # 设施
        self.reviews = info.reviews
        self.review_rating = info.review_rating
        self.review_A = info.review_A
        self.review_B = info.review_B
        self.review_C = info.review_C
        self.review_D = info.review_D

        if info.instant_bookable == "t":
            self.instant_bookable = 1
        elif info.instant_bookable == "f":
            self.instant_bookable = 0
        else:
            print("Un support instant bookable: %s" % info.instant_bookable)
            exit(-1)
        # self.instant_bookable = None

        if info.has_price():
            self.__has_price = True
            self.__price = info.get_price()
        else:
            self.__has_price = False
            self.__price = None

        return

    def has_price(self):
        return self.__has_price

    def get_price(self):
        assert self.has_price()
        return self.__price


def get_neighbor_tokenizer(infos: list[BookingInfo]) -> (dict, dict):
    """
    输入BookingInfo的列表，返回一个序列化的标准。
    :param infos: BookingInfo s
    :return: (token2str, str2token)
    """
    strs = list()
    for info in infos:
        strs.append(info.neighbor)

    return get_strs_tokenizer(strs=strs)


def get_strs_tokenizer(strs: list[str]) -> (dict, dict):
    """
    输入string的列表，表示属性。
    :param strs: string list.
    :return: (token2str, str2token)
    """
    token2str = dict()
    str2token = dict()

    for s in strs:
        _, token2str, str2token = strs_to_tokens([s], token2str=token2str, str2token=str2token)

    return token2str, str2token

