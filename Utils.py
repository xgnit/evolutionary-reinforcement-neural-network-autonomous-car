from PIL import Image, ImageDraw


class RectUtils:
    # useful for wall blocks, but not for the car, cause the dot product of the car's vertices might
    # result in something equals a very small number but not zero
    @staticmethod
    def sort_vertice(vertice):
        def vector(vertice1, vertice2):
            return [vertice2[0] - vertice1[0], vertice2[1] - vertice1[1]]
        def valid_vector(vec):
            return True if vec[0] and vec[1] else False
        def around_null(res):
            return True if abs(res) < 1 else False

        v1, v2, v3, v4 = vertice
        vec1 = vector(v1, v2)
        vec2 = vector(v3, v4)
        if valid_vector(vec1) and valid_vector(vec2) and 0 == vec1[0]*vec2[1] - vec1[1]*vec2[0]:
            vec1 = vector(v1, v2)
            vec2 = vector(v2, v3)
            if valid_vector(vec1) and valid_vector(vec2) and 0 == vec1[0] * vec2[0] + vec1[1] * vec2[1]:
                return [v1, v2, v3, v4]
            else:   return [v1, v2, v4, v3]
        else:
            vec1 = vector(v1, v3)
            vec2 = vector(v3, v4)
            if valid_vector(vec1) and valid_vector(vec2) and 0 == vec1[0] * vec2[0] + vec1[1] * vec2[1]:
                return [v1, v3, v4, v2]
            else:   return [v1, v3, v2, v4]



class ImageUtils:

    @staticmethod
    def make_image(i, j):
        return Image.new("RGB", (i, j), "white")

    @staticmethod
    def draw_rect(image, vertice, color=(200,200,200), outline=None):
        # im_ = image.load()
        ImageDraw.Draw(image).polygon(vertice, fill=color, outline=outline)

    @staticmethod
    def draw_car(image, vertice):
        v1, v2, v3, v4 = vertice

        ImageDraw.Draw(image).polygon(vertice, fill=None, outline=(200, 0,0))

        img = Image.open('car.png')
        img = img.rotate(45)
        img = img.resize((50, 50)).convert("RGBA")
        image.paste(img, (0, 0), mask=img)

        img = Image.open('car.png')
        img = img.rotate(45)
        img = img.resize((100, 100)).convert("RGBA")
        image.paste(img, (0, 100), mask=img)

        image.show()


def test_draw_car():
    dst = Image.new('RGB', (200, 200), 'blue')

    ImageUtils.draw_car(dst, ((0,0), (50,0), (50,50), (0,50)))



def test_sort_vertice():
    rect1 = [[0,0],[0,1],[1,0],[1,1]]
    rect2 = [[0,0],[1,1],[1,0],[1,0]]
    rect3 = [[0,0],[1,0],[0,1],[1,1]]
    print(RectUtils.sort_vertice(rect1))
    print(RectUtils.sort_vertice(rect2))
    print(RectUtils.sort_vertice(rect3))

def test_draw_rect():
    tmp = ImageUtils.make_image(500, 500)
    ImageUtils.draw_rect(tmp, ((0, 0), (0, 100), (100, 100), (100, 0)), color=(0, 200, 0))
    tmp.show()

if __name__ == "__main__":


    test_draw_car()

    # test_draw_rect()
    # test_sort_vertice()

