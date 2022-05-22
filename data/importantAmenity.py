class ImportantAmenity:
    def __init__(self):
        self.short_names = {
            "tv": 0,
            "parking": 1,
            "washer": 2,
            "cooking": 3,
            "coffee": 4,
            "wifi": 5,
            "wi-fi": 5,
            "kitchen": 6,
            "air conditioning": 7,
            "air conditioner": 7,
            "hair dryer": 8,
            "dryer": 9,
            "iron": 10,
            "gym": 11,
            "pool": 12,
            "smart": 13,
            "play": 14,
            "ps4": 15,
            "game console": 15,
            "nintendo": 15,
            "bathtub": 16
        }
        self.length = 17

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.short_names[item]
