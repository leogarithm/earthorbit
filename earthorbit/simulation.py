import numpy as np
import matplotlib.image as mpimg
import arrow
from typing import TypeVar, Generic

import math
import random
import json
import sys
import time

from earthorbit.orbit import Orbit
from earthorbit.orbitingobjects import SunOrbit, MoonOrbit
from earthorbit.maths import Maths
from earthorbit.timeconversion import TimeConversion

TypeVar_NumPy3DArray = TypeVar("NumPy 3D array")
TypeVar_NumPy3x3Matrix = TypeVar("NumPy 3x3 matrix")
TypeVar_DateTime = TypeVar("Arrow")
TypeVar_NumPyArray = TypeVar("NumPy array")
TypeVar_Orbit = TypeVar("earthorbit.orbit.Orbit")

class Simulation:
    """
        Class used to create a simulation of the trajectory of the satellite from its orbit.

        Attributes
        ----------
        start_epoch: Arrow object
            Start epoch of simulation
        simulation_duration: int
            Duration of simulation [s]
        time_step: int
            Time step of simulation [s]
        sun_events: list
            List containing dictionaries, each entry corresponds to an event where Sun is visible from satellite
        sun_events: list
            List containing dictionaries, each entry corresponds to an event where Moon is visible from satellite
        gs_events: dict
            Keys are names of groundstations. Each value is a list containing dictionaries, each entry corresponds to an event where the actual groundstation is visible from satellite
        roi_events: dict
            Keys are names of ROI. Each value is a list containing dictionaries, each entry corresponds to an event where the satellite is above the actual ROI

        Methods
        -------
        __init__()
            create simulation
        size_simu_MB()
            Returns the (approximative) space the simulation takes in memory, in megabytes (MB, or Mo in french)
    """
    def __init__(self, orbit: TypeVar_Orbit, start_epoch: TypeVar_DateTime, simulation_duration: int, sat_name: str, list_gs: list, list_roi: list, time_step=1, print_progress=False):
        """
            Simulation object

            :param orbit: Orbit of satellite
            :param start_epoch: Epoch corresponding to beggining of simulation
            :param simulation_duration: Duration of simulation [s]
            :param sat_name: Satellite name
            :param list_gs: List containing dictionaries. Each dictionary contains infos for gs. GPS position in [°], visibility angle in [rad]
            :param list_roi: List containing the .bmp file names used to create each ROI
            :param time_step: Time step of simulation. Warning: should not be set to less than one!
            :param print_progress: if 'True', writes in the console progress of simulation computation
            :type sat_gp_dict: dict
            :type start_epoch: Arrow object
            :type simulation_duration: int
            :type sat_name: str
            :type list_gs: list
            :type list_rois: list
            :type time_step: float
            :type print_progress: bool
        """
        start_chrono = time.process_time()

        self.start_epoch = start_epoch
        self.stop_epoch = self.start_epoch.shift(seconds=simulation_duration)
        self.simulation_duration = simulation_duration
        self.gs = list_gs
        self.roi = []

        for roi_fn in list_roi: # converts the huge matrix containing pixels of image in a list of coordinates corresponding to area
            if roi_fn.endswith(".bmp"):
                roi_name = roi_fn.replace(".bmp", "") # get the name of the file (it will be the ROI name inside script)
                img_as_matrix = mpimg.imread(roi_fn) # read the image using Matplotlib, as a matrix
                r = img_as_matrix[:,:,0]
                g = img_as_matrix[:,:,1]
                bool_roi = r != g
                r = {
                    "name": roi_name,
                    "img": bool_roi
                }
                self.roi.append(r)

        self.body_radius = { # [m]
            "earth": 6371000.0,
            "sun": 696340000.0,
            "moon": 1737100.0
        }

        self.dist_from_earth = { # [m]
            "moon": 384400000.0,
            "sun": 149600000000.0
        }

        self.time_step = time_step

        self.satellite = orbit
        self.sun = SunOrbit(start_epoch) # orbit for Sun at beggining of simulation
        self.moon = MoonOrbit(start_epoch) # orbit for Moon at beggining of simulation

        self.semi_angular_size = {
            "earth": np.arcsin(self.body_radius["earth"]/self.satellite.orbital_elements["semi_major_axis"]),
            "sun": np.arcsin(self.body_radius["sun"]/self.dist_from_earth["sun"]),
            "moon": np.arcsin(self.body_radius["moon"]/self.dist_from_earth["moon"])
        }

        self.total_steps = int(simulation_duration/self.time_step)
        
        self.timestamps = []

        gs_visi_values = {} # handling successive visibility values for event computation
        gs_pos_gcrs = {}
        self.gs_events = {} # events for each groundstation

        roi_visi_values = {} # handling successive visibility values for event computation
        self.roi_events = {} # events for each region of interest

        for el in self.gs: # initialising dict for gs
            gs_name = el["name"]
            gs_visi_values[gs_name] = False
            self.gs_events[gs_name] = []
        
        for el in self.roi: # initialising dict for roi
            roi_name = el["name"]
            roi_visi_values[roi_name] = False
            self.roi_events[roi_name] = []
        
        sun_visi_value = False # handling successive visibility values for event computation
        moon_visi_value = False # same huh
        self.sun_events = []
        self.moon_events = []

        nb_bars = 64 # variable for printing in console
        if print_progress:
            print("Simulation for '{}' beggining:".format(sat_name))
            print("░"*nb_bars, end="", flush=True)

        for step in range(0, self.total_steps): # computation for every frame (length of 1s if step not modified)
            if print_progress:
                if step%3600 == 0:
                    bars_prog = int(step/self.total_steps*nb_bars)
                    txt_prog = "█"*bars_prog
                    print("\r" + txt_prog, end="", flush=True)

            epoch = self.start_epoch.shift(seconds=step*self.time_step)

            sat_posvel_gcrs_dict = self.satellite.pos_vel_gcrs(epoch)
            sat_pos_gcrs = sat_posvel_gcrs_dict["position"]
            sat_vel_gcrs = sat_posvel_gcrs_dict["velocity"]
            sat_pos_itrs = self.satellite.gcrs2itrs(sat_pos_gcrs, epoch)

            sat_pos_itrs_sphe = Maths.rectangular2spherical(sat_pos_itrs)
            r = sat_pos_itrs_sphe[0] # [m]
            lon = np.rad2deg(sat_pos_itrs_sphe[1]) # [°]
            lat = np.rad2deg(sat_pos_itrs_sphe[2]) # [°]
            sat_pos_gps = np.array([r, lon, lat]) 

            sun_pos_gcrs = self.sun.pos_gcrs(epoch)
            moon_pos_gcrs = self.moon.pos_gcrs(epoch)

            sun_visi = self.check_visibility_star(sat_pos_gcrs, sun_pos_gcrs, self.semi_angular_size["sun"])
            moon_visi = self.check_visibility_star(sat_pos_gcrs, moon_pos_gcrs, self.semi_angular_size["moon"])

            self.computing_events(self.sun_events, sun_visi_value, sun_visi, step, "sun")
            self.computing_events(self.moon_events, moon_visi_value, moon_visi, step, "moon")
            sun_visi_value = sun_visi
            moon_visi_value = moon_visi

            for el in self.gs:
                gs_name = el["name"]
                gs_gps = np.array([el["distance"], el["lon"], el["lat"]])
                gs_itrs = Orbit.gps2itrs(gs_gps)
                last_visi = gs_visi_values[gs_name]
                curr_visi = self.check_gs_visibility(sat_pos_itrs, gs_itrs, el["semi_angle_visibility"])
                self.computing_events(self.gs_events[gs_name], last_visi, curr_visi, step, gs_name) # updating list of events...
                gs_visi_values[gs_name] = curr_visi # overwrite last visibility
                gs_pos_gcrs[gs_name] = Orbit.itrs2gcrs(gs_itrs, epoch)
            
            for el in self.roi:
                roi_name = el["name"]
                last_visi = roi_visi_values[roi_name]
                curr_visi = self.check_above_roi(sat_pos_gps, el["img"])
                self.computing_events(self.roi_events[roi_name], last_visi, curr_visi, step, roi_name)
                roi_visi_values[roi_name] = curr_visi # overwrite last visi

            timestamp = {
                "sat_pos_gcrs": sat_pos_gcrs,
                "sat_pos_itrs": sat_pos_itrs,
                "sat_pos_gps": sat_pos_gps,
                "sat_vel_gcrs": sat_vel_gcrs,
                "moon_pos_gcrs": moon_pos_gcrs,
                "moon_visibility": moon_visi,
                "sun_pos_gcrs": sun_pos_gcrs,
                "sun_visibility": sun_visi,
                "gs_visibilities": gs_visi_values,
                "gs_pos_gcrs": gs_pos_gcrs,
                "roi_visibilities": roi_visi_values
            }

            self.timestamps.append(timestamp)
        
        self.computation_duration = time.process_time() - start_chrono # time ellapsed for program to compute all the timestamps [s]

        if print_progress:
            print("\nDone! Time ellapsed during computation: {} seconds".format(self.computation_duration))

    def time2step(self, time: int) -> int:
        """
            Returns the step corresponding to given time (time [s] from beggining of simulation)

            :param time: Time from beggining of simulation
            :type time: int
            :return: Corresponding step
            :rtype: int
        """
        step = round(time/self.time_step)
        return np.clip(step, 0, self.total_steps - 1)
        
    def epoch2step(self, epoch: TypeVar_DateTime) -> int:
        """
            Returns the step corresponding to given epoch

            :param epoch: Epoch
            :type epoch: Arrow object
            :return: Corresponding step
            :rtype: int
        """
        time_sec = epoch.timestamp - self.start_epoch.timestamp
        return self.time2step(time_sec)

    def step2unixepoch(self, step: int) -> int:
        """
            From a step of simulation, returns the corresponding unixepoch [s]

            :param step: Step
            :type step: int
            :return: Corresponding unixepoch
            :rtype: int
        """
        return self.start_epoch.timestamp + int(step*self.time_step)
    
    def get_closest_timestamp(self, epoch: TypeVar_DateTime) -> dict:
        """
            From given epoch, returns the closest timestamp of simulation, a dictionary containing all the current infos of simulation (sat pos/vel, sun/moon pos, current visible gs/roi)

            :param epoch: Epoch
            :type epoch: Arrow object
            :return: Closest timestamp
        """
        step = self.epoch2step(epoch)
        return self.timestamps[step]
    
    def is_epoch_in_simu(self, epoch: TypeVar_DateTime) -> bool:
        """
            Checks if given epoch is inside time interval of simulation

            :param epoch: Epoch
            :type epoch: Arrow object
            :return: True if epoch is inside the interval of simulation
            :rtype: bool
        """
        start = int(self.start_epoch.timestamp)
        return True if (start <= epoch.timestamp <= start + self.simulation_duration) else False

    def check_visibility_star(self, sat_pos: TypeVar_NumPy3DArray, star_pos: TypeVar_NumPy3DArray, star_angle: float) -> TypeVar_NumPyArray:
        """
            From a position in GCRS frame, checks if object (at given position in GCRS) is visible (aka not hidden by Earth)

            :param sat_pos: Position of satellite in GCRS frame
            :type sat_pos: NumPy 3D Array
            :param star_pos: Position of object in GCRS frame
            :type star_pos: NumPy 3D Array
            :param star_angle: Semi angular size of given object
            :type star_angle: float
            :return: True if satellite can see the object
            :rtype: bool
        """
        from_sat = -1*sat_pos
        from_sat_star = from_sat + star_pos

        pdscal = np.dot(from_sat, from_sat_star)

        if (pdscal <= 0): # visible for sure
            return True
        else:
            prod_len = np.linalg.norm(from_sat)*np.linalg.norm(from_sat_star)
            cos_angle = pdscal/prod_len
            angle = np.arccos(cos_angle)
            sas_earth = self.semi_angular_size["earth"]

            if (angle > sas_earth + star_angle): # visible
                return True
            elif (angle < sas_earth): # visible
                return False
            else: # rise/set
                return bool((angle - sas_earth)/star_angle)
    
    def check_gs_visibility(self, sat_itrs: TypeVar_NumPy3DArray, gs_itrs: TypeVar_NumPy3DArray, gs_semi_angle_visibility: float) -> bool:
        """
            From a position in ITRS frame, checks if groundstation (position given in ITRS too) is visible (aka not hidden by Earth)

            :param sat_itrs: Position of satellite in ITRS frame
            :type sat_itrs: NumPy 3D Array
            :param gs_itrs: Position of object in ITRS frame
            :type gs_itrs: NumPy 3D Array
            :param gs_semi_angle_visibility: Semi cone angle of visibility for groundstation
            :type star_angle: float
            :return: True if satellite can see the groundstation
            :rtype: bool
        """
        from_gs = sat_itrs - gs_itrs
        pdscal = np.dot(gs_itrs, from_gs)

        if (pdscal <= 0): # not visible
            return False
        else:
            prod_len = np.linalg.norm(gs_itrs)*np.linalg.norm(from_gs)
            cos_angle = pdscal/prod_len
            angle = np.arccos(cos_angle)

            if (angle > gs_semi_angle_visibility): # still not visible
                return False
            else:
                return True
    
    def check_above_roi(self, sat_gps: TypeVar_NumPy3DArray, roi_mat: TypeVar_NumPyArray) -> bool:
        """
            Checks if satellite is directly above ROI

            :param sat_gps: Position of satellite in GPS coordinates
            :type sat_gps: NumPy 3D Array
            :param roi_mat: Matrix corresponding to ROI coordinates
            :type roi_mat: Numpy 180x360 Array
            :return: True if satellite is above ROI
            :rtype: bool
        """
        # computing the position of satellite on the 180x360 map grid
        xpos = int(np.floor(sat_gps[1]) + 180)
        ypos = int(np.floor(sat_gps[2]) + 90)
        matpos = np.zeros(roi_mat.shape)
        matpos[ypos, xpos] = 1
        isabove = matpos*roi_mat
        return bool(np.any(isabove))
    
    def computing_events(self, list_events: list, last_value: bool, current_value: bool, step: int, name_event: str):
        """
            Method that computes events for any object visibility.
            If an object was not visible last step and is for this new step: creates a event dictionary in given list events.
            If an object was visible last step and is not anymore for this new step: ends the event by writing the stop unixepoch.

            :param list_events: List of events of the object
            :type list_events: list
            :param last_value: Previous value for visibility
            :type last_value: bool
            :param current_value: Current value for visibility
            :type current_value: bool
            :param step: Current step of simulation
            :type step: int
            :param name_event: Name of the event
            :type name_event: str
        """
        if (last_value + current_value)%2 != 0: # stop or beg event
            if current_value: # beg
                list_events.append(
                    {
                        "name": name_event,
                        "start_unixepoch": self.step2unixepoch(step),
                        "stop_unixepoch": False
                    }
                )
            else: # end
                ev = list_events[-1]
                ev["stop_unixepoch"] = self.step2unixepoch(step)
                ev["duration_sec"] = ev["stop_unixepoch"] - ev["start_unixepoch"]
        else: # continue
            if step == self.total_steps - 1 and len(ev) > 0: # if last step, end last simu
                ev = list_events[-1]
                if ev["stop_unixepoch"] == False:
                    ev["stop_unixepoch"] = self.stop_epoch.timestamp
                    ev["duration_sec"] = ev["stop_unixepoch"] - ev["start_unixepoch"]
    
    def size_simu_MB(self) -> float:
        """
            returns the APPROXIMATIVE size of the list containing all of the timestamps for simulation in megabytes (MB)

            :return: Size of simulation in MB
            :rtype: float
        """
        samples = 32
        tot_size_byte = 0
        for i in range(0, samples):
            ts = random.choice(self.timestamps)
            tot_size_byte += sys.getsizeof(ts)
        return len(self.timestamps)*tot_size_byte/samples*1e-6