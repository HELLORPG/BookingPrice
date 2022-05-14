# 用来表示房屋预定信息

class BookingInfo:
    def __init__(self):
        """
        创建一个空白的实例
        """
        self.description = None
        self.neighbor = None
        self.latitude = None
        self.longitude = None
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

    def _price(self):
        if self.__has_price:
            return self.price
        else:
            print("This info has no self.price.")
            exit(-1)
