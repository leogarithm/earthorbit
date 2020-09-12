import numpy as np
import arrow
from typing import TypeVar, Generic

import math 
import json

from earthorbit.maths import Maths
from earthorbit.timeconversion import TimeConversion

TypeVar_NumPy3DArray = TypeVar("NumPy 3D array")
TypeVar_NumPy3x3Matrix = TypeVar("NumPy 3x3 matrix")
TypeVar_DateTime = TypeVar("Arrow")

class Orbit:
    """
    Class used to handle orbits. When created, contains all the information to describe the ellipse and calculate the satellite position over time.

    Attributes
    ----------
    orbital_elements: object
        eccentricity, inclination, right ascension of ascending node, argument of periaster, mean motion, mean anomaly, semi major axis, semi minor axis
    epoch: Arrow
        Arrow time. Represents when the orbital elements of the orbit are relevent, when they have been measured.

    Methods
    -------
    __init__()
        Creates orbit
    
    from_celestrak_json()
        Instantiate object from Celestrak requested string
    
    from_tle()
        Instantiate object from TLE string
    
    from_state_vectors()
        Instantiate object from position and velocity vectors, at given epoch
    
    get_mean_anomaly()
        Get mean anomaly (M) on the ellipse, at given epoch
    
    get_true_anomaly()
        Get true anomaly (ν) on the ellipse, at given epoch
    
    pos_gcrs()
        Get position [x, y, z] of satellite in GCRS frame (units: meters), at given epoch
    
    pos_vel_gcrs()
        Get position [x, y, z] of satellite in GCRS frame (units: meters), and velocity [vx, vy, vz] in GCRS frame (units: meters per seconds), at given epoch
    
    pos_gps()
        Get position [alt, lon, lat] (units: [meters, radians, radians]) of satellite in ITRS frame in polar coordinates (aka: GPS coordinates)
    """

    MU_EARTH = 398600441800000.0 # [m^3/s^2]

    def __init__(self, epoch: TypeVar_DateTime, eccentricity: float, inclination: float, ra_of_asc_node: float, arg_of_pericenter: float, mean_motion: float, mean_anomaly: float, name: str) -> None:
        """
        Instantiate object.

        :param epoch: correspoding to the epoch (aka. the date) of the orbital informations
        :param eccentricity: eccentricity value
        :param inclination: inclination value (units: radians)
        :param ra_of_asc_node: right ascension of ascending node value (units: radians)
        :param arg_of_pericenter: argument of periaster value (units: radians)
        :param mean_motion: mean motion value (units: radians per seconds)
        :param mean_anomaly: mean anomaly value (units: radians)
        :param name: name of satellite
        :type epoch: Arrow object
        :type eccentricity: float
        :type inclination: float
        :type ra_of_asc_node: float
        :type arg_of_pericenter: float
        :type mean_motion: float
        :type mean_anomaly: float
        :type name: string
        """

        self.epoch = epoch
        self.name = name

        self.orbital_elements = {
            "eccentricity": eccentricity,
            "inclination": inclination,
            "ra_of_asc_node": ra_of_asc_node,
            "arg_of_pericenter": arg_of_pericenter,
            "mean_motion": mean_motion,
            "mean_anomaly": mean_anomaly,
            "semi_major_axis": math.pow(Orbit.MU_EARTH/(mean_motion*mean_motion), 1/3)
        }

        a = self.orbital_elements["semi_major_axis"]
        e = eccentricity
        ee = e*e
        eee = ee*e

        self.trueanomaly_formula_factors = {
            1: 2*e - 0.25*eee,
            2: 5/4*ee,
            3: 13/12*eee
        }

        self.r_factor = a*(1 - ee)

        i_c = math.cos(inclination)
        i_s = math.sin(inclination)
        r_c = math.cos(ra_of_asc_node)
        r_s = math.sin(ra_of_asc_node)
        a_c = math.cos(arg_of_pericenter)
        a_s = math.sin(arg_of_pericenter)

        self.gcrs2orbitplane = np.array([
            [r_c, r_s, 0], 
            [-r_s*i_c, r_c*i_c, i_s], 
            [r_s*i_s, -r_c*i_s, i_c]]
        )
        self.orbitplane2gcrs = np.linalg.inv(self.gcrs2orbitplane)
        self.rotation_orbitplane = np.array([
            [a_c, -a_s, 0],
            [a_s, a_c, 0],
            [0, 0, 0]
        ])

        self.orbital_elements["time_at_periaster"] = epoch.timestamp - mean_anomaly/mean_motion # [s]
        self.orbital_elements["semi_minor_axis"] = self.orbital_elements["semi_major_axis"]*math.sqrt(1 - ee) # [m]
        
    
    def compute_true_anomaly(self, mean_anomaly: float) -> float:
        """
        Compute true anomaly (nu) for the given mean anomaly (M)

        :param mean_anomaly: Mean anomaly, in radians [rad]
        :type mean_anomaly: float
        :returns: Corresponding true anomaly, in radians [rad]
        :rtype: float
        """
        return (mean_anomaly 
               + self.trueanomaly_formula_factors[1]*math.sin(mean_anomaly)
               + self.trueanomaly_formula_factors[2]*math.sin(2*mean_anomaly)
               + self.trueanomaly_formula_factors[3]*math.sin(3*mean_anomaly))
    
    def get_mean_anomaly(self, epoch: TypeVar_DateTime) -> float:
        """
        Compute mean anomaly (M) at given epoch

        :param epoch: correspoding epoch
        :type epoch: Arrow object
        :returns: Corresponding mean anomaly, in radians [rad]
        :rtype: float
        """
        n = self.orbital_elements["mean_motion"]
        t = self.orbital_elements["time_at_periaster"]

        M = n*(epoch.timestamp - t)
        return M%Maths.TWOPI
    
    def get_true_anomaly(self, epoch: TypeVar_DateTime) -> float:
        """
        Compute true anomaly (nu) at given epoch

        :param epoch: correspoding epoch
        :type epoch: Arrow object
        :returns: Corresponding true anomaly. units: radians [rad]
        :rtype: float
        """
        M = self.get_mean_anomaly(epoch)
        return self.compute_true_anomaly(M)
    
    def true_anomaly2gcrs(self, true_anomaly: float) -> TypeVar_NumPy3DArray:
        """
        Compute position (in GCRS rectangular frame) corresponding to the given true anomaly (nu)

        :param true_anomaly: True anomaly
        :type true_anomaly: float
        :returns: Corresponding position in GCRS rectangular Frame (NumPy 3D vector). units: meters ([m], [m], [m])
        :rtype: NumPy 3D array
        """
        # position of satellite in orbit plane (with z = 0)
        r = self.r_factor/(1 + self.orbital_elements["eccentricity"]*np.cos(true_anomaly))
        x_orbitplane = r*math.cos(true_anomaly)
        y_orbitplane = r*math.sin(true_anomaly)

        pos_plane = self.rotation_orbitplane.dot([x_orbitplane, y_orbitplane, 0])

        pos_gcrs = self.orbitplane2gcrs.dot(pos_plane)
        return pos_gcrs
    
    def pos_gcrs(self, epoch: TypeVar_DateTime) -> TypeVar_NumPy3DArray:
        """
        Compute position (in GCRS rectangular frame) corresponding at given epoch

        :param epoch: corresponding epoch
        :type epoch: Arrow object
        :returns: Corresponding position in GCRS rectangular Frame (NumPy 3D vector). units: meters ([m], [m], [m])
        :rtype: NumPy 3D array
        """
        nu = self.get_true_anomaly(epoch)
        return self.true_anomaly2gcrs(nu)
    
    def pos_vel_gcrs(self, epoch: TypeVar_DateTime, dt=0.001) -> dict:
        """
        Compute position and velocity (in GCRS rectangular frame) corresponding at given epoch

        :param epoch: corresponding epoch
        :param dt: Time precision for numerical derivation. The lesser, the more precise will be the velocity
        :type epoch: Arrow object
        :type dt: float
        :returns: Dictionary containing position and velocity in GCRS Frame (NumPy 3D vector). units: meters ([m], [m], [m]), ([m/s], [m/s], [m/s]), 
        :rtype: dict
        """
        unixepoch = epoch.timestamp
        next_epoch = arrow.get(unixepoch + dt).to("utc")
        pos = self.pos_gcrs(epoch)
        next_pos = self.pos_gcrs(next_epoch)
        vel = (next_pos - pos)/dt
        return {
            "position": pos,
            "velocity": vel
        }
    
    def pos_itrs(self, epoch: TypeVar_DateTime) -> TypeVar_NumPy3DArray:
        """
        Compute position (in ITRS rectangular frame) corresponding at given epoch

        :param epoch: corresponding epoch
        :type epoch: Arrow object
        :returns: Dictionary containing position in ITRS Frame (NumPy 3D vector). units: meters ([m], [m], [m]), 
        :rtype: NumPy 3D array
        """
        gcrs = self.pos_gcrs(epoch)
        return Orbit.gcrs2itrs(gcrs, epoch)
    
    def pos_gps(self, epoch: TypeVar_DateTime) -> TypeVar_NumPy3DArray:
        """
        Compute position (in ITRS spherical frame, aka. GPS coordinates) corresponding at given epoch

        :param epoch: corresponding epoch
        :type epoch: Arrow object
        :return: NumPy 3D vector of position in GPS coordinates (radius, longitude, latitude). units: ([m], [°], [°])
        :rtype: NumPy 3D array
        """
        pos_itrs = self.pos_itrs(epoch)
        pos_gps_rad = Maths.rectangular2spherical(pos_itrs)

        lon = np.rad2deg(pos_gps_rad[1])
        lat = np.rad2deg(pos_gps_rad[2])

        pos_gps = np.array([pos_gps_rad[0], lon, lat])
        return pos_gps

    @staticmethod
    def rotationmatrix_z(angle: float) -> TypeVar_NumPy3x3Matrix:
        """
        Compute a rotation matrix around the z axis for a given angle

        :param angle: Angle to rotate
        :type angle: float
        :return: matrix corresponding to the rotation of the given angle around the z axis
        :rtype: NumPy 3x3 matrix
        """
        return np.array([
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1]
            ]
        )
    
    @staticmethod
    def stl0(epoch: TypeVar_DateTime) -> float:
        """
        Compute the Sideral Time of Latitude 0
        (algorithm taken here: https://fr.wikipedia.org/wiki/Temps_sid%C3%A9ral#Calcul_de_l'heure_sid%C3%A9rale)

        :param epoch: corresponding epoch
        :type epoch: Arrow object
        :return: STL0 at given epoch. units: radians [rad]
        :rtype: float
        """
        j2000 = TimeConversion.unixepoch2j2000(epoch.timestamp)
        stl0hour = 18.697374558 + 24.06570982441908*j2000
        stl0rad = stl0hour*math.pi/12
        return stl0rad%Maths.TWOPI
    

    @staticmethod
    def gcrs2itrs(cartesian_coord: TypeVar_NumPy3DArray, epoch: TypeVar_DateTime) -> TypeVar_NumPy3DArray:
        """
        Converts the cartesian coordinates from GCRS rectangular frame to ITRS rectangular frame, at given epoch

        :param cartesian_coord: vector coordinates in GCRS rectangular frame to be converted
        :param epoch: corresponding epoch
        :type cartesian_coord: NumPy 3D vector
        :type epoch: Arrow object
        :return: coordinates converted to ITRS rectangular frame
        :rtype: NumPy 3D array
        """
        angle = Orbit.stl0(epoch)
        rotationmatrix = Orbit.rotationmatrix_z(-angle)
        return rotationmatrix.dot(cartesian_coord)
    
    @staticmethod
    def itrs2gcrs(cartesian_coord: TypeVar_NumPy3DArray, epoch: TypeVar_DateTime) -> TypeVar_NumPy3DArray:
        """
        Converts the cartesian coordinates from ITRS rectangular frame to GCRS rectangular frame, at given epoch

        :cartesian_coord: coordinates in ITRS rectangular frame to be converted
        :param epoch: corresponding epoch
        :type cartesian_coord: NumPy 3D vector
        :type epoch: Arrow object
        :return: coordinates converted to ITRS rectangular frame
        :rtype: NumPy 3D array
        """
        angle = Orbit.stl0(epoch)
        rotationmatrix = Orbit.rotationmatrix_z(angle)
        return rotationmatrix.dot(cartesian_coord)
    
    @staticmethod
    def gps2itrs(gps_coords: TypeVar_NumPy3DArray) -> TypeVar_NumPy3DArray:
        """
        Converts GPS coordinates, from ITRS spherical frame to ITRS rectangular frame

        :param gps_coords: coordinates in ITRS spherical frame to be converted (units: [m], [°], [°])
        :type cartesian_coord: NumPy 3D vector
        :return: coordinates converted to ITRS rectangular frame
        :rtype: NumPy 3D array
        """
        alt = gps_coords[0]
        lon = np.deg2rad(gps_coords[1])
        lat = np.deg2rad(gps_coords[2])
        return Maths.spherical2rectangular(np.array([alt, lon, lat]))
    
    @staticmethod
    def gps2gcrs(gps_coords: TypeVar_NumPy3DArray, epoch: TypeVar_DateTime) -> TypeVar_NumPy3DArray:
        """
        Converts GPS coordinates, from ITRS spherical frame to GCRS rectangular frame, at given epoch

        :param gps_coords: coordinates in ITRS spherical frame to be converted (units: [m], [°], [°])
        :param epoch: corresponding epoch
        :type cartesian_coord: NumPy 3D vector
        :type epoch: Arrow object
        :return: coordinates converted to ITRS rectangular frame
        :rtype: NumPy 3D array
        """
        to_itrs = Orbit.gps2itrs(gps_coords)
        to_gcrs = Orbit.itrs2gcrs(to_itrs, epoch)
        return to_gcrs


    @classmethod
    def from_tle(cls, tle: str) -> object:
        """
        Creates an instance of Orbit object, using TLE (Two-Line Elements) string. 
        String may contain a title line. If not, this is not a problem, a name will be added automatically.
        Conventions used can be found here: https://en.wikipedia.org/wiki/Two-line_element_set

        :param tle: TLE for the orbit, at given time. Three lines separated with '\n' (slash n)
        :type tle: str
        :returns: Orbit object
        """
        if tle.endswith("\n"): # if ends with break line, remove the last one
            tle = tle[0:len(tle) - 2]

        tle = tle.replace("\r", "")
        splited = tle.split("\n")

        name = "UNNAMED SATELLITE"
        first = ""
        second = ""

        if len(splited) > 2: # therefore first line is for satellite name
            name = splited[0]
            first = splited[1]
            second = splited[2]
        else:
            first = splited[0]
            second = splited[1]

        epoch_year = int(first[18:20])
        epoch_decimal_day = float(first[20:32])
        epoch_day = math.floor(epoch_decimal_day)
        epoch_decimaltime_inday = epoch_decimal_day - epoch_day
        epoch_decimaltime_hours = 24*epoch_decimaltime_inday
        epoch_hours = math.floor(epoch_decimaltime_hours)
        epoch_decimaltime_inhours = epoch_decimaltime_hours - epoch_hours
        epoch_decimaltime_minutes = 60*epoch_decimaltime_inhours
        epoch_minutes = math.floor(epoch_decimaltime_minutes)
        epoch_decimaltime_inminutes = epoch_decimaltime_minutes - epoch_minutes
        epoch_decimaltime_seconds = 60*epoch_decimaltime_inminutes
        epoch_seconds = math.floor(epoch_decimaltime_seconds)

        time = str(epoch_year) + "-" + str(epoch_day) + " " + str(epoch_hours) + ":" + str(epoch_minutes) + ":" + str(epoch_seconds)

        epoch = arrow.get(time, "YY-DDD H:m:s")

        inclination = float(second[8:16])
        raan = float(second[17:24])
        eccentricity = float("0." + second[26:33])
        argp = float(second[34:42])
        mean_anomaly = float(second[43:51])
        mean_motion = float(second[52:65])

        return cls(epoch, eccentricity, inclination, raan, argp, mean_motion, mean_anomaly, name)
    
    @classmethod
    def from_state_vectors(cls, pos: TypeVar_NumPy3DArray, vel: TypeVar_NumPy3DArray, epoch: TypeVar_DateTime, name="UNNAMED SATELLITE") -> object:
        """
        Creates an instance of Orbit object, using both position and velocity vectors at given epoch

        :param pos: vector of position, in GCRS rect frame, at given epoch. units: ([m], [m], [m])
        :param vel: vector of velocity, in GCRS rect frame, at given epoch. units: ([m/s], [m/s], [m/s])
        :param epoch: corresponding epoch
        :type pos: NumPy 3D array
        :type vel: NumPy 3D array
        :type epoch: Arrow object
        :returns: Orbit object
        """
        x = np.array([1, 0, 0])
        z = np.array([0, 0, 1])

        kinetic = np.cross(pos, vel)
        kinetic_sq = kinetic.dot(kinetic)
        pos_dir = Maths.normalize_vect(pos)
        ecc_vec = np.cross(vel, kinetic)/Orbit.MU_EARTH - pos_dir
        minusy_ellipse = np.cross(kinetic, ecc_vec)*(-1)

        e = np.linalg.norm(ecc_vec) # [1]
        ee = e*e
        eee = ee*e
        eeee = eee*e
        i = Maths.angle_vects(z, kinetic) # [rad]
        nu = Maths.angle_vects(pos_dir, ecc_vec)
        raan = Maths.angle_vects(x, minusy_ellipse) # [rad]
        argp = Maths.angle_vects(minusy_ellipse, ecc_vec) # [rad]
        a = kinetic_sq/Orbit.MU_EARTH/(1 - e*e)
        aaa = a*a*a
        n = np.sqrt(Orbit.MU_EARTH/aaa) # [rad/s]
        M = (nu 
            - 2*e*np.sin(nu) 
            + (3/4*ee + 1/8*eeee)*np.sin(2*nu) 
            - 1/3*eee*np.sin(3*nu) 
            + 5/32*eeee*np.sin(4*nu))

        return cls(epoch, np.rad2deg(i), np.rad2deg(raan), np.rad2deg(argp), n*86400/np.TWO_PI, np.rad2deg(M), "idk")
    
    @classmethod
    def from_celestrak_json(cls, stringified_json: str) -> object:
        """
        Creates an instance of Orbit object, using a .json string given by Celestrak.

        :param stringified_json: .json string containing orbital elements from a Celestrak request
        :type stringified_json: str
        :returns: Orbit object
        """
        json_loaded = json.loads(stringified_json.replace("\r\n", ""))
        celestrak_json = json_loaded[0] if isinstance(json_loaded, list) else json_loaded # many results in request. therefore take first one

        name = celestrak_json["OBJECT_NAME"]
        epoch = arrow.get(celestrak_json["EPOCH"]).to("utc")
        eccentricity = celestrak_json["ECCENTRICITY"]
        inclination = np.deg2rad(celestrak_json["INCLINATION"])
        ra_of_asc_node = np.deg2rad(celestrak_json["RA_OF_ASC_NODE"])
        arg_of_pericenter = np.deg2rad(celestrak_json["ARG_OF_PERICENTER"])
        mean_anomaly = np.deg2rad(celestrak_json["MEAN_ANOMALY"])
        mean_motion = celestrak_json["MEAN_MOTION"]*Maths.TWOPI/86400
        return cls(epoch, eccentricity, inclination, ra_of_asc_node, arg_of_pericenter, mean_motion, mean_anomaly, name)
