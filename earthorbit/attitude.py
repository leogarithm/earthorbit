from typing import TypeVar, Generic

import numpy as np
import arrow
import pyquaternion as quat

from earthorbit.simulation import Simulation
from earthorbit.maths import Maths

TypeVar_Quaternion = TypeVar("Quaternion (pyquaternion)")
TypeVar_NumPy3DArray = TypeVar("NumPy 3D array")
TypeVar_NumPy3x3Matrix = TypeVar("NumPy 3x3 matrix")
TypeVar_DateTime = TypeVar("Arrow")
TypeVar_NumPyArray = TypeVar("NumPy array")
TypeVar_eoSimulation = TypeVar("earthorbit.simulation.Simulation")

class Attitude:
    """
        Class used to handle commands and rotations to be send to satellite during simulation.

        Attributes
        ----------
        
        

        Methods
        -------
        __init__()
        push_command()
            Pushes a command into execution list. Therefore will be added to final generated command list
        del_command()
            Removes a command that was pushed.
        push_target()
            Pushes a direction target, in GCRS, into list of targets. When command generation, will be converted into corresponding quaternion rotation and will be added to final generated command list
        del_target()
            Removes a target that was pushed.
        push_target_antisun()
            Pushes a target, where the target is the opposite direction of Sun
        push_target_moon()
            Pushes a target, where the target is the direction of Moon
        push_target_nadir()
            Pushes a target, where the target is NADIR (aka pointing ground)
        push_target_limbs_ortho()
            Pushes a target, where the target is LIMBS (aka perpendicular to ground), orthogonal to velocity
        push_target_limbs_vel()
            Pushes a target, where the target is LIMBS (aka perpendicular to ground), colinear to velocity
        push_target_gs()
            Pushes a target, where the target is given groundstation
        rot_qsw()
            From a start direction and a final direction, returns the quaternion to make this rotation
        print_list_commands()
            Prints the list of commands and destinations given at initialisation
        generate_list_commands()
            Generates the list of commands to be sent to satellite, corresponding to all commands and rotations pushed
        
    """
    def __init__(self, simulation: TypeVar_eoSimulation, list_cmd: dict, list_dest: dict, start_quat: TypeVar_Quaternion, id_qsw_rot: tuple, av_rot_vel=0.2):
        """
            :param list_cmd: List of commands that can be executed by satellite
            :type list_cmd: dict
            :param list_dest: List of destinations where to send the commands to satellite
            :type list_dest: dict
            :param start_quat: Orientation quaternion at the beggining of simulation
            :type start_quat: pyquaternion object
            :param id_qsw_rot: Tuple containing id for rotation command and destination that corresponds to the command executed to make rotations in QSW frame
            :param av_rot_vel: average rotation velocity of satellite [deg/s]
            :param av_rot_vel: float
        """
        self.simulation = simulation
        self.list_cmd = Attitude.dict_strkey2intkey(list_cmd)
        self.list_dest = Attitude.dict_strkey2intkey(list_dest)
        self.id_qsw_rot_cmd, self.id_qsw_rot_dest = id_qsw_rot
        self.gcrs_target_request = {}
        self.cmd_request = {}
        self.push_target(self.simulation.start_epoch, start_quat) # first attitude at start

        self.generated_commands = []

        self.average_rotation_speed = np.deg2rad(av_rot_vel) # [rad/s]
    
    def push_target(self, epoch: TypeVar_DateTime, gcrs_target_quat: TypeVar_Quaternion):
        """
            Pushes desired target at given epoch. Returns the ID of this push.

            :param epoch: Epoch when the satellite has to be pointing to target
            :type epoch: Arrow object
            :param gcrs_target_quat: Orientation quaternion corresponding to the target
            :type epoch: pyquaternion object
            :return: push ID
            :rtype: str
        """
        idx = Attitude.gen_unique_id()
        if self.simulation.is_epoch_in_simu(epoch):
            self.gcrs_target_request[idx] = {
                "id": idx,
                "gcrs_target_quat": gcrs_target_quat,
                "epoch": epoch,
                "ts": epoch.timestamp
            }
            return idx
        else:
            raise ValueError("Given epoch '{}' is outside of simulation!".format(epoch.format()))
    
    def del_target(self, idx: str):
        """
            Removes a target previously pushed.

            :param idx: push ID
            :type idx: str
            :return: True if target successfully removed
            :rtype: bool
        """
        try:
            self.gcrs_target_request.pop(idx)
            return True
        except:
            raise

    def push_command(self, epoch: TypeVar_DateTime, id_cmd: int, id_dest: int, args: list):
        """
            Pushes desired command at given epoch. Returns the ID of this push.

            :param epoch: Epoch when the satellite has to execute the command
            :type epoch: Arrow object
            :param id_cmd: command ID (aka key of command inside commands dictionary)
            :param id_cmd: int
            :param id_dest: destination ID (aka key of destination inside destination dictionary)
            :param id_dest: int
            :return: push ID
            :rtype: str
        """
        cmd = ""
        dest = ""

        try:
            cmd = self.list_cmd[id_cmd] # dict
            dest = self.list_dest[id_dest] # str
            if len(args) != 0:
                cmd_args = cmd["args"]
                if len(cmd_args) != len(args):
                    raise ValueError("Incorrect number of arguments given! {} given, {} expected.".format(len(args), len(cmd_args)))
                for (a, t) in zip(args, cmd_args):
                    type_arg = getattr(np, t)
                    if np.can_cast(a, type_arg) == False:
                        raise TypeError("Type of '{}' is '{}' and should be '{}'!".format(a, type(a), t))
        except:
            raise IndexError("Incorrect command or destination ID!")
        
        idx = Attitude.gen_unique_id()
        if self.simulation.is_epoch_in_simu(epoch):
            self.cmd_request[idx] = {
                "id": idx,
                "cmd": cmd["name"],
                "dest": dest,
                "epoch": epoch,
                "ts": epoch.timestamp,
                "args": args
            }
            return idx
        else:
            raise ValueError("Given epoch '{}' is outside of simulation!".format(epoch.format()))

    def del_command(self, idx: str):
        """
            Removes a command previously pushed.

            :param idx: push ID
            :type idx: str
            :return: True if command successfully removed
            :rtype: bool
        """
        try:
            self.cmd_request.pop(idx)
            return True
        except:
            raise
    
    def push_target_moon(self, epoch: TypeVar_DateTime):
        """
            Pushes a target at given epoch. Direction is: Moon.

            :param epoch: Epoch when the satellite has to be pointing to target
            :type epoch: Arrow object
        """
        ts = self.simulation.get_closest_timestamp(epoch)
        moon_pos_gcrs = ts["moon_pos_gcrs"]
        sat_pos_gcrs = ts["sat_pos_gcrs"]
        target = moon_pos_gcrs - sat_pos_gcrs
        target_quat = Maths.get_orientation_quat(target)
        return self.push_target(epoch, target_quat)

    def push_target_antisun(self, epoch: TypeVar_DateTime):
        """
            Pushes a target at given epoch. Direction is: Sun opposite.

            :param epoch: Epoch when the satellite has to be pointing to target
            :type epoch: Arrow object
        """
        ts = self.simulation.get_closest_timestamp(epoch)
        sun_pos_gcrs = ts["sun_pos_gcrs"]
        sat_pos_gcrs = ts["sat_pos_gcrs"]
        target = sat_pos_gcrs - sun_pos_gcrs
        target_quat = Maths.get_orientation_quat(target)
        return self.push_target(epoch, target_quat)

    def push_target_nadir(self, epoch: TypeVar_DateTime):
        """
            Pushes a target at given epoch. Direction is: NADIR.

            :param epoch: Epoch when the satellite has to be pointing to target
            :type epoch: Arrow object
        """
        ts = self.simulation.get_closest_timestamp(epoch)
        target = -1*ts["sat_pos_gcrs"]
        target_quat = Maths.get_orientation_quat(target)
        return self.push_target(epoch, target_quat)

    def push_target_limbs_ortho(self, epoch: TypeVar_DateTime):
        """
            Pushes a target at given epoch. Direction is: LIMBS and orthogonal to velocity.

            :param epoch: Epoch when the satellite has to be pointing to target
            :type epoch: Arrow object
        """
        ts = self.simulation.get_closest_timestamp(epoch)
        sat_pos_gcrs = ts["sat_pos_gcrs"]
        x = sat_pos_gcrs/np.linalg.norm(sat_pos_gcrs)
        sat_vel_gcrs = ts["sat_vel_gcrs"]
        y = sat_vel_gcrs/np.linalg.norm(sat_vel_gcrs)
        target = np.cross(x, y)
        target_quat = Maths.get_orientation_quat(target)
        return self.push_target(epoch, target_quat)

    def push_target_limbs_vel(self, epoch: TypeVar_DateTime):
        """
            Pushes a target at given epoch. Direction is: LIMBS and colinear to velocity.

            :param epoch: Epoch when the satellite has to be pointing to target
            :type epoch: Arrow object
        """
        ts = self.simulation.get_closest_timestamp(epoch)
        target = ts["sat_vel_gcrs"]
        target_quat = Maths.get_orientation_quat(target)
        return self.push_target(epoch, target_quat)

    def push_target_gs(self, epoch: TypeVar_DateTime, gs_name: str):
        """
            Pushes a target at given epoch. Direction is: Given groundstation.

            :param epoch: Epoch when the satellite has to be pointing to target
            :type epoch: Arrow object
            :param gs_name: Name of groundstation
            :type gs_name: str
        """
        ts = self.simulation.get_closest_timestamp(epoch)
        gs_pos_gcrs = ts["gs_pos_gcrs"][gs_name]
        sat_pos_gcrs = ts["sat_pos_gcrs"]
        target = gs_pos_gcrs - sat_pos_gcrs
        target_quat = Maths.get_orientation_quat(target)
        return self.push_target(epoch, target_quat)

    def rot_qsw(self, start_quat: TypeVar_NumPy3DArray, target_quat: TypeVar_NumPy3DArray, unix_target: int):
        """
            From a direction and a target, returns the quaternion corresponding in QSW frame of satellite

            :param start_quat: Orientation quaternion corresponding to start direction
            :type start_quat: pyquaternion object
            :param target_quat: Orientation quaternion corresponding to target direction
            :type target_quat: pyquaternion object
            :param unix_target: Unixepoch when the satellite has to be pointing to target [s]
            :type unix_target: int
            :return: Dictionary containing the infos for rotation, and quaternion in QSW frame
        """
        rot_in_gcrs = target_quat*start_quat.inverse # to get the rotation to make from start to target, we rotate to x axis AND apply orientation quaternion of target

        x = np.array([1, 0, 0])

        start_dir = start_quat.rotate(x)
        target_dir = target_quat.rotate(x)

        angle_torot = Maths.angle_vects(start_dir, target_dir)
        time_torot = angle_torot/self.average_rotation_speed
        unix_start = round(unix_target - time_torot)

        ts_start = self.simulation.get_closest_timestamp(arrow.get(unix_start))

        x_qsw = ts_start["sat_pos_gcrs"]
        rot_quat_sat = Maths.get_orientation_quat(x_qsw)

        rot_quat = rot_in_gcrs*rot_quat_sat.inverse # to get the rotation quaternion in QSW frame, we rotate to x axis AND apply orientation of target

        return {
            "qsw_quat": rot_quat,
            "gcrs_start_dir": start_dir,
            "gcrs_target_dir": target_dir,
            "unix_start": unix_start,
            "unix_stop": unix_target,
            "duration": time_torot
        }

    def print_list_commands(self):
        """
            Prints the list of commands and destinations given at initialisation into console.
        """
        print("###################")
        print("COMMANDS:")
        for (idx, cmd) in self.list_cmd.items():
            print("{}: '{}'".format(idx, cmd))
        print("------------")
        print("DESTINATIONS:")
        for (idx, dest) in self.list_dest.items():
            print("{}: '{}'".format(idx, dest))
        print("###################")
    
    def generate_list_commands(self, safe_angle=np.deg2rad(30)):
        """
            Generates the list of commands to be sent to satellite, corresponding to all commands and rotations pushed.
            Makes some verifications, like checks for BBQ and for rotations overlaps.
            :param safe_angle: Maximum angle that the satellite and Sun can make. If angle is greater, rotation not considered as safe.
            :type safe_angle: float
            :return: True if generated without problems. Returns a list of warnings if problems were detected, such as BBQ of overlaps.
            :rtype: bool|list
        """

        alerts = []
        commands = []

        for cmd in self.cmd_request.values():
            commands.append(Attitude.dict_cmd(
                cmd["cmd"],
                cmd["dest"],
                cmd["ts"],
                cmd["args"]
            ))
        
        rotations = self.generate_rotations()
        for rot in rotations:
            commands.append(Attitude.dict_cmd(
                self.list_cmd[self.id_qsw_rot_cmd]["name"],
                self.list_dest[self.id_qsw_rot_dest],
                rot["unix_start"],
                list(rot["qsw_quat"])
            ))

        ## SAFE CHECKS ##
        bbq = self.bbq_check(rotations, safe_angle)
        overlap = self.overlap_check(rotations)
        ##             ##

        if bbq != True:
            for rot in bbq:
                txt = "BBQ occurs between {} and {}".format(
                    arrow.get(rot["unix_start"]),
                    arrow.get(rot["unix_stop"])
                )
                alerts.append(txt)
        
        if overlap != True:
            for ts in overlap:
                txt = "Overlap occurs around {}".format(
                    arrow.get(ts)
                )
                alerts.append(txt)

        self.generated_commands = commands

        return alerts if len(alerts) != 0 else True

    @staticmethod
    def dict_cmd(fx: str, dest: str, unixepoch: int, args: list):
        return {
            "fx": fx,
            "dest": dest,
            "unixepoch": unixepoch,
            "args": args
        }

    def generate_rotations(self):
        """
            returns the list of QSW rotations for all targets entered
        """
        targ_req = list(self.gcrs_target_request.values())
        last_target = targ_req[0]
        all_targets = targ_req.copy()
        all_targets.pop(0)

        rot_list = []

        for target in all_targets:
            start_quat = last_target["gcrs_target_quat"]
            target_quat = target["gcrs_target_quat"]
            unix_target = target["ts"]
            generated_rot = self.rot_qsw(start_quat, target_quat, unix_target)
            rot_list.append(generated_rot)

            last_target = target
        
        return rot_list

    def bbq_check(self, rot_list: list, safe_angle: float):
        """
            check for bbq
            :param rot_list: list of rotations generated with 'rot_qsw'
            :param safe_angle: safe angle for bbq [rad]
        """
        bbq_alert = []

        for rot in rot_list:
            x = rot["gcrs_start_dir"]
            y = rot["gcrs_target_dir"]
            z = np.cross(x, y)

            t = arrow.get(rot["unix_start"])
            ts = self.simulation.get_closest_timestamp(t)
            s = ts["sun_pos_gcrs"]
            s = Maths.normalize_vect(s)

            alert = False

            angle_norm_plane = Maths.angle_vects(z, s)
            
            if angle_norm_plane > Maths.HALFPI - safe_angle:
                if Maths.angle_vects(x, s) < safe_angle or Maths.angle_vects(y, s) < safe_angle:
                    alert = True
                else:
                    a = np.cross(z, x) 
                    b = np.cross(y, z)
                    if np.dot(s, a) > 0 and np.dot(s, b) > 0:
                        alert = True
            
            if alert:
                bbq_alert.append(rot)
        
        return bbq_alert if len(bbq_alert) != 0 else True
        

    def overlap_check(self, rot_list: list):
        """
            check if two rotations overlap
        """
        ts_list = []

        for rot in rot_list:
            ts_list.append((rot["unix_start"], rot["unix_stop"]))
        
        ts_list.sort() # sort by ascending start time
        mat_ts = np.array(ts_list)
        start_vec = mat_ts[:,0]
        start_vec_r = np.delete(start_vec, 0)
        stop_vec = mat_ts[:,1]
        stop_vec_r = np.delete(stop_vec, -1)

        overlaps_vec = stop_vec_r < start_vec_r

        return start_vec_r[overlaps_vec].tolist() if overlaps_vec.sum() != 0 else True

    @staticmethod
    def gen_unique_id() -> str:
        return str(arrow.utcnow().timestamp) + str(np.random.randint(0, 2147483647))
    
    @staticmethod
    def dict_strkey2intkey(d):
        """
            Convert the stri key into int key if possible
        """
        newd = {}
        for k, v in d.items():
            try:
                intk = int(k)
                newd[intk] = v
            except ValueError:
                print("Given dict has '{}' key which cannot be converted to 'int'!".format(k))
        return newd
