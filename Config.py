






class Config:

    @staticmethod
    def car_width_base():
        return 20
    @classmethod
    def car_length_base(cls):
        return 2 * cls.car_width_base()

    @classmethod
    def path_width(cls):
        return 2 * cls.car_width_base() + cls.car_width_base()//2

    @classmethod
    def map_size(cls):
        return cls.map_scaler() * cls.map_base()

    @staticmethod
    def map_scaler():
        return 2
    @staticmethod
    def map_base():
        return 256
    @staticmethod
    def path_gray_rbg():
        return (169, 169, 169)

    @staticmethod
    def bg_rbg():
        return (220, 220, 220)

    @classmethod
    def grid_size(cls):
        return cls.map_size() // cls.path_width()

    @staticmethod
    def wall_rbg():
        return (64, 64, 64)

    @classmethod
    def used_map_size(cls):
        return cls.path_width() * cls.grid_size()