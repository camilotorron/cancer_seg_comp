class Bbox:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def __init__(self, x_min: int, x_max: int, y_min: int, y_max: int):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
