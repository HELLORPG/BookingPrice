# data package 用来对数据进行处理

__all__ = ["bookingInfo", "opFile"]

from .bookingInfo import BookingInfo    # 使用类似的代码，可以减少引用BookingInfo Class时所需的字段层次，from data import BookingInfo 就可以。
from .opFile import DataFile
