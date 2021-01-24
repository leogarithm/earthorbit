from typing import TypeVar, Generic
import numpy as np
import arrow

from earthorbit.simulation import Simulation
from earthorbit.maths import Maths

TypeVar_NumPy3DArray = TypeVar("NumPy 3D array")
TypeVar_NumPy3x3Matrix = TypeVar("NumPy 3x3 matrix")
TypeVar_DateTime = TypeVar("Arrow")
TypeVar_NumPyArray = TypeVar("NumPy array")
TypeVar_eoAttitude = TypeVar("earthorbit.attitude.Attitude")

class Macro():
    """
        Class to stack a series of specific commands/orientations to be pushed a multiple of times.
    """
    def __init__(self, list_cmd: list):
        """
            :param list_cmd: list of dictionaries containing 'fx' (function), and 'exe_time' (int), and 'extra_args' (list)
        """
        self.list_cmd = list_cmd

    def apply_macro(self, att: TypeVar_eoAttitude, epoch: TypeVar_DateTime):
        """
            Applies the series of commands to 'earthorbit.attitude.Attitude' object at given epoch.
            Each command will be executed at given epoch + 'exe_time' seconds, where 'exe_time' is a parameter given for each command in initialisation of this class.

            :param att: Object to apply commands
            :type att: earthorbit.attitude.Attitude
            :param epoch: Epoch when to apply series
            :type epoch: Arrow object
        """
        for cmd in self.list_cmd:
            exe_time = cmd["exe_time"]
            fx = cmd["fx"]
            exe_epoch = epoch.shift(seconds=exe_time)
            extra_args = cmd["extra_args"]
            fx(att, exe_epoch, *extra_args) # execute append command for given attitude
    
    def apply_macro_periodic(self, att: TypeVar_eoAttitude, start_epoch: TypeVar_DateTime, stop_epoch: TypeVar_DateTime, frequency: int):
        """
            Applies this macro to the given Attitude object, at given frequency between start epoch and stop epoch given.

            :param frequency: [s]
            :type frequency: int
            :param start_epoch: start epoch
            :type start_epoch: Arrow object
            :param stop_epoch: stop epoch
            :type stop_epoch: Arrow object
        """
        duration = stop_epoch.timestamp - start_epoch.timestamp
        for t in range(0, duration, frequency):
            epoch = start_epoch.shift(seconds=t)
            self.apply_macro(att, epoch)
    
    def apply_macro_event(self, att: TypeVar_eoAttitude, event: dict, relative_exe_time=0.0):
        """
            Applies the series of commands to 'earthorbit.attitude.Attitude' object once during the given event.
            Execution time is computed with 'relative_exe_time' given. 
            For instance, if 'relative_exe_time' = 0 the macro will be executed at the beggining of the event.
            if 'relative_exe_time' = 1 the macro fill be executed at the end of the event.
            if relative_exe_time = 0.5 the macro will be executed at the middle of the event...
        """
        start_event = arrow.get(event["start_unixepoch"])
        duration_event = event["stop_unixepoch"] - event["start_unixepoch"]
        if duration_event > 0:
            relative_exe_time = np.clip(relative_exe_time, 0, 1)
            exe_time = start_event.timestamp + relative_exe_time*duration_event
            self.apply_macro(att, start_event.shift(seconds=exe_time))
    
    def apply_macro_periodic_event(self, att: TypeVar_eoAttitude, frequency: int, event: dict):
        """
            Applies this macro to the given Attitude object, at given frequency during the event given.
        """
        start_event = arrow.get(event["start_unixepoch"])
        stop_event = arrow.get(event["stop_unixepoch"])
        if stop_event.timestamp > start_event.timestamp:
            self.apply_macro_periodic(att, start_event, stop_event, frequency)