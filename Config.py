






class Config:

    @staticmethod
    def car_width_base():
        return 15
    @classmethod
    def car_length_base(cls):
        return 2 * cls.car_width_base()

    @classmethod
    def path_width(cls):
        return 2 * cls.car_width_base() + 10

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
    def gray_rbg():
        return (169, 169, 169)