# 该文件并不是该项目的主线运行流程
# 只是用于测试一些函数等尝试性探索

from data.utils import strs_to_tokens
from data import InfoFile, BookingInfo


if __name__ == '__main__':
    infos = InfoFile("./dataset/train.csv").csv_to_booking_info()

    info_t2s, info_s2t = dict(), dict()
    for info in infos:
        info.amenities, info_t2s, info_s2t = strs_to_tokens(info.amenities, token2str=info_t2s, str2token=info_s2t)

    print(infos[5])
    print(info_s2t)
