from PIL import Image, ImageDraw


class ImageUtils():
    @staticmethod
    def make_image(i, j):
        return Image.new("RGB", (i, j), "white")

