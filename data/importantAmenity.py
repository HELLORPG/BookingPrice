class ImportantAmenity:
    def __init__(self):
        self.short_names = {
            "tv": 0,
            "parking": 1,
            "washer": 2
        }
        self.length = 3

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.short_names[item]
