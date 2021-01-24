import math
from typing import TypeVar, Generic

import numpy as np
import pyquaternion

NumPy3DArray = TypeVar("NumPy 3D array")
NumPy4DArray = TypeVar("NumPy 4D array")
NumPy3x3Matrix = TypeVar("NumPy 3x3 matrix")
DateTime = TypeVar("datetime object")

class Maths:
    TWOPI = 2*math.pi
    HALFPI = 0.5*math.pi

    @staticmethod
    def normalize_vect(v: NumPy3DArray) -> NumPy3DArray:
        """
        Returns the normalized vector (Euclidian norm)

        :param v: NumPy 3D vector
        :return: normalized NumPy 3D vector
        """
        norm = np.linalg.norm(v)
        return v/norm if norm != 0 else v
    
    @staticmethod
    def angle_vects(v: NumPy3DArray, w: NumPy3DArray) -> float:
        """
        Returns the angle, in radians, between the two NumPy 3D vectors given

        :param v: NumPy 3D vector
        :param w: NumPy 3D vector
        :return: Angle, in radians, between v and w
        """
        if np.all(v != w):
            d = w.dot(v)
            vnorm = np.linalg.norm(v)
            wnorm = np.linalg.norm(w)
            prod_norms = vnorm*wnorm
            cos_angle = np.clip(d/prod_norms, -1, 1)
            return np.arccos(cos_angle)
        else:
            return 0

    @staticmethod
    def rectangular2spherical(rect_coords: NumPy3DArray) -> NumPy3DArray:
        """
        Converts a rectangular coordinates into spherical coordinates

        :param rect_coords: Numpy 3D vector, rectangular coordinates to be converted 
        :return: Numpy 3D vector, coordinates converted into spherical. (radius, longitude [rad], latitude [rad])
        """

        r = np.linalg.norm(rect_coords)
        lon = Maths.HALFPI # [-pi/2, pi/2]

        if (rect_coords[0] == 0):
            lon *= 1 if rect_coords[1] > 0 else -1
        else:
            lon = math.atan(rect_coords[1]/rect_coords[0])
            lon += math.pi if rect_coords[0] < 0 and rect_coords[1] > 0 else 0
            lon -= math.pi if rect_coords[0] < 0 and rect_coords[1] < 0 else 0

        lat = np.arcsin(rect_coords[2]/r) # [-pi/2, pi/2]

        return np.array([r, lon, lat])

    @staticmethod
    def spherical2rectangular(sphe_coords: NumPy3DArray) -> NumPy3DArray:
        """
        Converts a spherical coordinates into rectangular coordinates

        :param rect_coords: Numpy 3D vector, spherical coordinates to be converted (radius, longitude [rad], latitude [rad])
        :return: Numpy 3D vector, coordinates converted into rectangular. 
        """

        r = sphe_coords[0]
        lon = sphe_coords[1]
        lat = sphe_coords[2]
        cos_lat = math.cos(lat)

        return np.array(
            [
                r*math.cos(lon)*cos_lat,
                r*math.sin(lon)*cos_lat,
                r*math.sin(lat)
            ]
        )
    
    @staticmethod
    def truncate_decimals(nb: float, remaining_decimals: int) -> float:
        """
        Take a float and remove extra decimals.
        Example: truncate_decimals(0.1234, 2) returns 0.12

        :param nb: float to be truncated
        :param remaining_decimals: the number of remaining decimals wanted for the number
        :returns: the number truncated
        """
        tenpowpos = 1
        for i in range(0, remaining_decimals):
            tenpowpos *= 10
        tenpowneg = 1/tenpowpos
        return float(int(nb*tenpowpos))*tenpowneg
    
    @staticmethod
    def get_rot_quat(start: NumPy3DArray, target: NumPy3DArray):
        """
            returns the rotation quaternion to make in order to go from 'start' to 'target'
            q = w + ix + jy + kz
        """
        angle = Maths.angle_vects(start, target)
        dir_rot = np.cross(start, target)
        return pyquaternion.Quaternion(axis=dir_rot, radians=angle)
    
    @staticmethod
    def get_orientation_quat(vec: NumPy3DArray):
        """
            From a 3D direction 'vec' in space, returns its quaternion orientation, aka the rotation from x axis to 'vec'.
        """
        return Maths.get_rot_quat(np.array([1, 0, 0]), vec)
