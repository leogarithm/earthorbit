import arrow

from earthorbit.timeconversion import TimeConversion

class Vts:
    """
        Methods to generate CIC files for VTS (CNES software)
    """
    @staticmethod
    def generate_commands(attitude: object) -> str:
        """
            Creates text for an event file, prompts the commands to be executed.
        """
        creationdate = str(arrow.utcnow()).replace("+00:00", "")
        base = """CIC_MEM_VERS = 1.0
COMMENT 'COMMANDS' Generated by 'earthorbit' Python library (author: Léo Giroud)
CREATION_DATE = {}
ORIGINATOR = EARTHORBIT PYTHON LIBRARY

META_START

USER_DEFINED_PROTOCOL = NONE
USER_DEFINED_CONTENT = EVENTS
USER_DEFINED_SIZE = 1
USER_DEFINED_TYPE = STRING
USER_DEFINED_UNIT = [n/a]
TIME_SYSTEM = UTC

META_STOP

""".format(creationdate)
        for cmd in attitude.generated_commands:
            txt = [cmd["fx"] + "(" + str(cmd["args"]) + ") at " + cmd["dest"]]
            base += Vts.generate_line(arrow.get(cmd["unixepoch"]), txt)
        return base

    @staticmethod
    def generate_events(dict_events: dict):
        """
            Creates text for an event file, prompts events when visible/hidden.
        """
        creationdate = str(arrow.utcnow()).replace("+00:00", "")
        base = """CIC_MEM_VERS = 1.0
COMMENT 'GENERIC EVENTS' Generated by 'earthorbit' Python library (author: Léo Giroud)
CREATION_DATE = {}
ORIGINATOR = EARTHORBIT PYTHON LIBRARY

META_START

USER_DEFINED_PROTOCOL = NONE
USER_DEFINED_CONTENT = EVENTS
USER_DEFINED_SIZE = 1
USER_DEFINED_TYPE = STRING
USER_DEFINED_UNIT = [n/a]
TIME_SYSTEM = UTC

META_STOP

""".format(creationdate)

        for e in dict_events:
            start_event = arrow.get(e["start_unixepoch"])
            stop_event = arrow.get(e["stop_unixepoch"])
            name_event = e["name"]
            base += Vts.generate_line(start_event, [name_event + " visible"])
            base += Vts.generate_line(stop_event, [name_event + " hidden"])
        
        return base

    @staticmethod
    def generate_position(simulation: object) -> str:
        """
            Creates text for an event file, for position and velocity
        """
        creationdate = str(arrow.utcnow()).replace("+00:00", "")
        base = """CIC_OEM_VERS = 2.0
COMMENT 'POSITION' Generated by 'earthorbit' Python library (author: Léo Giroud)
CREATION_DATE = {}
ORIGINATOR = EARTHORBIT PYTHON LIBRARY

META_START

OBJECT_NAME = UNKNOWN
OBJECT_ID = UNKNOWN
CENTER_NAME = UNDEFINED
REF_FRAME = EME2000
TIME_SYSTEM = UTC

META_STOP

""".format(creationdate)
        for i in range(0, simulation.total_steps):
            unixepoch = simulation.step2unixepoch(i)
            ts = simulation.timestamps[i]
            pos = ts["sat_pos_gcrs"]*1e-3 # [km]
            vel = ts["sat_vel_gcrs"]*1e-3 # [km/s]
            els = [str(pos[0]), str(pos[1]), str(pos[2]), str(vel[0]), str(vel[1]), str(vel[2])]
            base += Vts.generate_line(arrow.get(unixepoch), els)
        return base
    
    @staticmethod
    def generate_orientation(attitude: object) -> str:
        """
            Creates text for an event file, for orientation quaternions
        """
        creationdate = str(arrow.utcnow()).replace("+00:00", "")
        base = """CIC_AEM_VERS = 1.0
COMMENT 'QUATERNIONS' Generated by 'earthorbit' Python library (author: Léo Giroud)
CREATION_DATE  = {}
ORIGINATOR     = EARTHORBIT PYTHON LIBRARY

META_START

OBJECT_NAME = UNKNOWN
OBJECT_ID = UNKNOWN

REF_FRAME_A = ICRF
REF_FRAME_B = SC_BODY_1
ATTITUDE_DIR = A2B
TIME_SYSTEM = UTC
ATTITUDE_TYPE = QUATERNION

META_STOP

""".format(creationdate)
        targets = attitude.gcrs_target_request
        for target in targets.values():
            epoch = target["epoch"]
            quat = target["gcrs_target_quat"]
            (w, x, y, z) = list(quat)
            base += Vts.generate_line(epoch, [str(w), str(x), str(y), str(z)])
        return base
    
    @staticmethod
    def generate_line(epoch: object, els: list) -> str:
        els_str = "	".join(els)
        return """{} {}
""".format(str(epoch).replace("+00:00", ""), els_str)