# 将bookingInfo类别编码之后得到的特征
# 便于后续的处理

from math import isnan

from data import BookingInfo
from data.utils import strs_to_tokens
from data.bookingInfo import Bathroom, BathroomType


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
        self.latitude = nan_handle(info.latitude)       # 纬度
        self.longitude = nan_handle(info.longitude)     # 经度
        self.type = None
        self.accommodates = nan_handle(info.accommodates)
        self.bathrooms = bathroom_to_token(info.bathrooms)
        self.bedrooms = nan_handle(info.bedrooms)
        self.amenities = None  # 设施
        self.reviews = nan_handle(info.reviews)
        self.review_rating = nan_handle(info.review_rating)
        self.review_A = nan_handle(info.review_A)
        self.review_B = nan_handle(info.review_B)
        self.review_C = nan_handle(info.review_C)
        self.review_D = nan_handle(info.review_D)

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


def nan_handle(value):
    """
    用来处理可能的NAN数值。
    :param value: value that may be nan.
    :return: if nan, None. else, value.
    """
    if isnan(value):
        return None
    else:
        return value


def bathroom_to_token(bathrooms: Bathroom) -> list:
    """
    根据Bathroom类，生成一个对应的Token。
    :param bathrooms:
    :return: list of value.
    """
    if bathrooms.type == BathroomType.MISSING:
        return None
    else:
        token = [0] * BathroomType.MISSING.value    # MISSING.vale表示了bathroom种类的个数
        token[bathrooms.type.value] = bathrooms.num
        return token


def get_neighbor_tokenizer(infos: list[BookingInfo]) -> (dict, dict):
    """
    输入BookingInfo的列表，返回一个neighbor的序列化标准。
    :param infos: BookingInfo s
    :return: (token2str, str2token)
    """
    strs = list()
    for info in infos:
        strs.append(info.neighbor)

    return get_strs_tokenizer(strs=strs)


def get_type_tokenizer(infos: list[BookingInfo]) -> (dict, dict):
    """
    输入BookingInfo的列表，返回一个type的序列化标准。
    :param infos: BookingInfo s
    :return: (token2str, str2token)
    """
    strs = list()
    for info in infos:
        strs.append(info.type)

    return get_strs_tokenizer(strs=strs)


def get_amenities_tokenizer(infos: list[BookingInfo]) -> (dict, dict):
    """
    输入BookingInfo的列表，返回一个amenities的序列化标准。
    :param infos: BookingInfo s
    :return: (token2str, str2token)
    """
    strs = list()
    for info in infos:
        for amenity in info.amenities:
            strs.append(amenity)

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


def infos_to_tokens(infos: list[BookingInfo]) -> (list[BookingToken], dict):
    """
    将BookingInfo类的列表转换成为BookingToken类的列表。
    :param infos: booking info list.
    :return: booking token list, tokenizer dict.
    """
    tokenizer = dict()  # 序列化的字典

    tokenizer["neighbor"], tokenizer["type"], tokenizer["amenities"] = dict(), dict(), dict()
    tokenizer["neighbor"]["token2str"], tokenizer["neighbor"]["str2token"] = get_neighbor_tokenizer(infos=infos)
    tokenizer["type"]["token2str"], tokenizer["type"]["str2token"] = get_type_tokenizer(infos=infos)
    tokenizer["amenities"]["token2str"], tokenizer["amenities"]["str2token"] = get_amenities_tokenizer(infos=infos)

    return infos_to_tokens_with_tokenizer(infos=infos, tokenizer=tokenizer)


def infos_to_tokens_with_tokenizer(infos: list[BookingInfo], tokenizer: dict) -> (list[BookingToken], dict):
    """
    将BookingInfo的列表转换成为BookingToken的列表，传入一个tokenizer来表示序列化的规则。
    :param infos: booking info list.
    :param tokenizer: some attribute tokenizer.
    :return: booking token list, and the tokenizer dict.
    """
    tokens = list()
    for info in infos:
        tokens.append(BookingToken(info=info))

    # 进行One Hot类的映射
    for i in range(0, len(tokens)):
        tokens[i].neighbor = [0] * len(tokenizer["neighbor"]["str2token"])
        tokens[i].neighbor[tokenizer["neighbor"]["str2token"][infos[i].neighbor]] = 1

        tokens[i].type = [0] * len(tokenizer["type"]["str2token"])
        tokens[i].type[tokenizer["type"]["str2token"][infos[i].type]] = 1

        tokens[i].amenities = [0] * len(tokenizer["amenities"]["str2token"])
        for amenity in infos[i].amenities:
            tokens[i].amenities[tokenizer["amenities"]["str2token"][amenity]] = 1

    return tokens, tokenizer


if __name__ == '__main__':
    from data import InfoFile
    all_infos = InfoFile("../dataset/split/train.csv").csv_to_booking_info()
    all_tokens, _ = infos_to_tokens(all_infos)
    print(all_tokens[0].amenities)
