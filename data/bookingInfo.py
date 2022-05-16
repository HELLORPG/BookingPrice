# 用来表示房屋预定信息

class BookingInfo:
    def __init__(self):
        """
        创建一个空白的实例
        """
        self.description = None
        self.neighbor = None
        self.latitude = None    # 纬度
        self.longitude = None   # 经度
        self.type = None
        self.accommodates = None
        self.bathrooms = None
        self.bedrooms = None
        self.amenities = None
        self.reviews = None
        self.review_rating = None
        self.review_A = None
        self.review_B = None
        self.review_C = None
        self.review_D = None
        self.instant_bookable = None

        # 用来表示标签，只有训练数据会有
        self.__has_price = False
        self.price = None

    def __init__(self, info: dict):
        """
        使用info字典来初始化实例
        :param info: booking info in dict.
        """
        self.description = info["description"]
        self.neighbor = info["neighbor"]
        self.latitude = info["latitude"]
        self.longitude = info["longitude"]
        self.type = info["type"]
        self.accommodates = info["accommodates"]
        self.bathrooms = info["bathrooms"]
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
            self.price = info["price"]

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
        info["bathrooms"] = self.bathrooms
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
            info["price"] = self.price

        return info

    def get_price(self):
        if self.__has_price:
            return self.price
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

