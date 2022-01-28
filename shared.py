from typing import Union
import numpy as np


class Vec2(np.ndarray):
    def __new__(cls, x: Union[int, float], y: Union[int, float]) -> "Vec2":
        return np.array((x, y)).view(cls)

    @property
    def x(self) -> Union[int, float]:
        return self[0]

    @property
    def y(self) -> Union[int, float]:
        return self[1]

    @x.setter
    def x(self, value: Union[int, float]):
        self[0] = value

    @y.setter
    def y(self, value: Union[int, float]):
        self[1] = value

    @property
    def magnitude(self) -> float:
        mag: float = np.sqrt(self.dot(self))
        return mag

    @property
    def angle(self) -> float:
        rad: float = np.arctan(self[1] / self[0]) if self[0] != 0 else np.pi / 2
        return np.rad2deg(rad)

    @staticmethod
    def direction(p1, p2) -> float:
        vec: Vector = p2 - p1
        return vec.angle

    @staticmethod
    def distance(p1, p2) -> float:
        vec: Vector = p2 - p1
        return vec.magnitude

    @property
    def normalized(self) -> "Vec2":
        mag = self.magnitude
        if mag == 0:
            return Vector(0, 0)

        return self * (1 / self.magnitude)
