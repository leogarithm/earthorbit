import numpy as np
import arrow
from typing import TypeVar, Generic

import math 

from earthorbit.maths import Maths
from earthorbit.timeconversion import TimeConversion
from earthorbit.orbit import Orbit

TypeVar_NumPy3DArray = TypeVar("NumPy 3D array")
TypeVar_NumPy3x3Matrix = TypeVar("NumPy 3x3 matrix")
TypeVar_DateTime = TypeVar("Arrow")


class SunOrbit(Orbit):
    """
        Instantiate a Orbit object containing the orbit of the Sun around the Earth.
        Orbital elements for this object taken from here: https://stjarnhimlen.se/comp/ppcomp.html#4

        :param epoch: correspoding to the epoch (aka. the date) of the orbital informations
        :type epoch: Arrow object
    """
    def __init__(self, epoch: TypeVar_DateTime):
        nb_days = TimeConversion.unixepoch2daysfrommillennium(epoch.timestamp)

        raan = 0
        i = np.deg2rad(23.4406) # [rad]
        argp = np.deg2rad(282.9404 + 4.70935e-5*nb_days) # [rad]
        e = 0.016709 - 1.151e-9*nb_days # [1]
        mean_motion = 1.9912e-7 # [rad/s]
        mean_anomaly = np.deg2rad(356.0470 + 0.9856002585*nb_days) # [rad]

        Orbit.__init__(self, epoch, e, i, raan, argp, mean_motion, mean_anomaly, "Sun")
    
class MoonOrbit(Orbit):
    def __init__(self, epoch: TypeVar_DateTime):
        """
            Instantiate a Orbit object containing the orbit of the Moon around the Earth.
            Orbital elements for this object taken from here: https://stjarnhimlen.se/comp/ppcomp.html#4

            :param epoch: correspoding to the epoch (aka. the date) of the orbital informations
            :type epoch: Arrow object
        """
        nb_days = TimeConversion.unixepoch2daysfrommillennium(epoch.timestamp)

        raan = np.deg2rad(125.1228 - 0.0529538083*nb_days) # [rad]
        i = 0.08980417133211624 # [rad]
        argp = np.deg2rad(318.0634 + 0.1643573223*nb_days) # [rad]
        e = 0.054900 # [1]
        mean_motion = 2.6698e-6 # [rad/s]
        mean_anomaly = np.deg2rad(115.3654 + 13.0649929509*nb_days) # [rad]

        Orbit.__init__(self, epoch, e, i, raan, argp, mean_motion, mean_anomaly, "Moon")
