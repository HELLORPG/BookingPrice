# 将bookingInfo类别编码之后得到的特征
# 便于后续的处理

from math import isnan, nan

from data import ImportantAmenity
from data import BookingInfo
from data.utils import strs_to_tokens, get_statistic, min_max
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
        self.amenities = None                           # 设施
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

        self.aline_feature = None

        return

    def has_price(self):
        return self.__has_price

    def get_price(self):
        assert self.has_price()
        return self.__price

    def get_aline_feature(self) -> list:
        return self.neighbor + [self.latitude] + [self.longitude] + self.type + [float(self.accommodates)] +\
            self.bathrooms + [self.bedrooms] + self.amenities + [float(self.reviews)] + [self.review_rating] +\
            [self.review_A] + [self.review_B] + [self.review_C] + [self.review_D] + [float(self.instant_bookable)]

    def build_aline_feature(self):
        self.aline_feature = self.get_aline_feature()


def norm_tokens(tokens: list[BookingToken], token_statistics: dict):
    """
    对所有的tokens进行norm。
    :param tokens:
    :param token_statistics:
    :return:
    """
    for i in range(0, len(tokens)):
        tokens[i].latitude = min_max(
            value=tokens[i].latitude,
            value_max=token_statistics["latitude"]["max"],
            value_min=token_statistics["latitude"]["min"]
        )

        tokens[i].longitude = min_max(
            value=tokens[i].longitude,
            value_max=token_statistics["longitude"]["max"],
            value_min=token_statistics["longitude"]["min"]
        )

        tokens[i].longitude = min_max(
            value=tokens[i].accommodates,
            value_max=token_statistics["accommodates"]["max"],
            value_min=token_statistics["accommodates"]["min"]
        )

        tokens[i].bedrooms = min_max(
            value=tokens[i].bedrooms,
            value_max=token_statistics["bedrooms"]["max"],
            value_min=token_statistics["bedrooms"]["min"]
        )

        tokens[i].accommodates = min_max(
            value=tokens[i].accommodates,
            value_max=token_statistics["accommodates"]["max"],
            value_min=token_statistics["accommodates"]["min"]
        )

        tokens[i].reviews = min_max(
            value=tokens[i].reviews,
            value_max=token_statistics["reviews"]["max"],
            value_min=token_statistics["reviews"]["min"]
        )

        tokens[i].review_rating = min_max(
            value=tokens[i].review_rating,
            value_max=token_statistics["review_rating"]["max"],
            value_min=token_statistics["review_rating"]["min"]
        )

        tokens[i].review_A = min_max(
            value=tokens[i].review_A,
            value_max=token_statistics["review_ABCD"]["max"],
            value_min=token_statistics["review_ABCD"]["min"]
        )

        tokens[i].review_B = min_max(
            value=tokens[i].review_B,
            value_max=token_statistics["review_ABCD"]["max"],
            value_min=token_statistics["review_ABCD"]["min"]
        )

        tokens[i].review_C = min_max(
            value=tokens[i].review_C,
            value_max=token_statistics["review_ABCD"]["max"],
            value_min=token_statistics["review_ABCD"]["min"]
        )

        tokens[i].review_D = min_max(
            value=tokens[i].review_D,
            value_max=token_statistics["review_ABCD"]["max"],
            value_min=token_statistics["review_ABCD"]["min"]
        )

        for j in range(0, len(tokens[i].bathrooms)):
            tokens[i].bathrooms[j] = min_max(
                value=tokens[i].bathrooms[j],
                value_max=token_statistics["bathrooms"]["max"],
                value_min=token_statistics["bathrooms"]["min"]
            )

    return tokens


def fix_loss_tokens(tokens: list[BookingToken], token_statistics: dict):
    """
    修正tokens中的缺失值。
    :param tokens:
    :param token_statistics:
    :return:
    """
    for i in range(0, len(tokens)):
        if tokens[i].latitude is None or tokens[i].longitude is None:
            tokens[i].latitude, tokens[i].longitude = token_statistics["latitude"]["mean"], token_statistics["longitude"]["mean"]
        if tokens[i].bedrooms is None:
            tokens[i].bedrooms = token_statistics["bedrooms"]["mean"]
        if tokens[i].bathrooms is None:
            tokens[i].bathrooms = [0.0] * BathroomType.MISSING.value
            for j in range(0, len(tokens[i].bathrooms)):
                tokens[i].bathrooms[j] = token_statistics["bathrooms"]["mean"]
        if tokens[i].accommodates is None:
            tokens[i].accommodates = token_statistics["accommodates"]["mean"]
        if tokens[i].review_rating is None:
            tokens[i].review_rating = token_statistics["review_rating"]["mean"]
        if tokens[i].review_A is None:
            tokens[i].review_A = token_statistics["review_ABCD"]["mean"]
        if tokens[i].review_B is None:
            tokens[i].review_B = token_statistics["review_ABCD"]["mean"]
        if tokens[i].review_C is None:
            tokens[i].review_C = token_statistics["review_ABCD"]["mean"]
        if tokens[i].review_D is None:
            tokens[i].review_D = token_statistics["review_ABCD"]["mean"]

    return tokens


def get_tokens_statistic(tokens: list[BookingToken]) -> dict:
    """
    返回tokens列表中，需要的统计数据。
    :param tokens:
    :return:
    """
    statistics = dict()

    latitudes = list()
    longitudes = list()
    reviews = list()
    bedrooms = list()
    bathrooms = list()
    accommodates = list()
    review_ABCDs = list()
    review_ratings = list()

    for token in tokens:
        latitudes.append(token.latitude)
        longitudes.append(token.longitude)
        reviews.append(token.reviews)
        bedrooms.append(token.bedrooms)
        bathrooms += [bathroom for bathroom in token.bathrooms] if token.bathrooms is not None else []
        accommodates.append(token.accommodates)
        review_ABCDs += [token.review_A, token.review_B, token.review_C, token.review_D]
        review_ratings.append(token.review_rating)

    statistics["review_ABCD"] = {
        "min": 0.0,
        "max": 10.0,
        "mean": get_statistic(review_ABCDs, mode="mean")
    }

    statistics["review_rating"] = {
        "min": 0.0,
        "max": 100.0,
        "mean": get_statistic(review_ratings, mode="mean")
    }

    statistics["latitude"] = {
        "min": get_statistic(latitudes, mode="min"),
        "max": get_statistic(latitudes, mode="max"),
        "mean": get_statistic(latitudes, mode="mean")
    }

    statistics["longitude"] = {
        "min": get_statistic(longitudes, mode="min"),
        "max": get_statistic(longitudes, mode="max"),
        "mean": get_statistic(longitudes, mode="mean")
    }

    statistics["reviews"] = {
        "min": get_statistic(reviews, mode="min"),
        "max": get_statistic(reviews, mode="max"),
        "mean": get_statistic(reviews, mode="mean")
    }

    statistics["bedrooms"] = {
        "min": get_statistic(bedrooms, mode="min"),
        "max": get_statistic(bedrooms, mode="max"),
        "mean": get_statistic(bedrooms, mode="mean")
    }

    statistics["bathrooms"] = {
        "min": get_statistic(bathrooms, mode="min"),
        "max": get_statistic(bathrooms, mode="max"),
        "mean": get_statistic(bathrooms, mode="mean")
    }
    statistics["accommodates"] = {
        "min": get_statistic(accommodates, mode="min"),
        "max": get_statistic(accommodates, mode="max"),
        "mean": get_statistic(accommodates, mode="mean")
    }

    return statistics


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
        token = [0.0] * BathroomType.MISSING.value    # MISSING.vale表示了bathroom种类的个数
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
    # tokenizer["amenities"]["token2str"], tokenizer["amenities"]["str2token"] = get_amenities_tokenizer(infos=infos)

    return infos_to_tokens_with_tokenizer(infos=infos, tokenizer=tokenizer)


def infos_to_tokens_with_tokenizer(infos: list[BookingInfo], tokenizer: dict) -> (list[BookingToken], dict):
    """
    将BookingInfo的列表转换成为BookingToken的列表，传入一个tokenizer来表示序列化的规则。
    :param infos: booking info list
    :param tokenizer: some attribute tokenizer.
    :return: booking token list, and the tokenizer dict.
    """
    tokens = list()
    for info in infos:
        tokens.append(BookingToken(info=info))

    # 进行One Hot类的映射
    for i in range(0, len(tokens)):
        tokens[i].neighbor = [0.0] * len(tokenizer["neighbor"]["str2token"])
        tokens[i].neighbor[tokenizer["neighbor"]["str2token"][infos[i].neighbor]] = 1.0

        tokens[i].type = [0.0] * len(tokenizer["type"]["str2token"])
        tokens[i].type[tokenizer["type"]["str2token"][infos[i].type]] = 1.0

        # tokens[i].amenities = [0.0] * len(tokenizer["amenities"]["str2token"])
        # for amenity in infos[i].amenities:
        #     tokens[i].amenities[tokenizer["amenities"]["str2token"][amenity]] = 1.0
        important_amenities = ImportantAmenity()
        tokens[i].amenities = [0.0] * len(important_amenities)
        for importance in important_amenities.short_names.keys():
            for amenity in infos[i].amenities:
                if amenity.lower().find(importance) != -1:
                    tokens[i].amenities[important_amenities[importance]] = 1.0

    return tokens, tokenizer


if __name__ == '__main__':
    from data import InfoFile
    all_infos = InfoFile("../dataset/split/train.csv").csv_to_booking_info()
    all_tokens, _ = infos_to_tokens(all_infos)
    # print(all_tokens[0].amenities)
    # for t in all_tokens:
    #     print(len(t.get_aline_feature()))

    # token = all_tokens[0]
    # for name, value in vars(token).items():
    #     print(name, value)
    #     print(type(value))

    statistics = get_tokens_statistic(all_tokens)
    all_tokens = fix_loss_tokens(all_tokens, statistics)
    all_tokens = norm_tokens(all_tokens, statistics)
    for token in all_tokens:
        token.build_aline_feature()
        print(token.amenities)
