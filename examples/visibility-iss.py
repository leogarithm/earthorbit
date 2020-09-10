"""
    visibility-iss.py
"""

from earthorbit.orbit import Orbit
from earthorbit.maths import Maths
import numpy as np
import arrow
import requests

eiffeltour_gps = np.array([34, 2.2942, 48.8582])
eiffeltour_itrs = Orbit.gps2itrs(eiffeltour_gps)
visibility_angle = np.deg2rad(20)

req = requests.get("https://celestrak.com/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE")
iss = Orbit.from_tle(req.text)

date = arrow.utcnow()

visible = False
for i in range(0, 20000):
    date = date.shift(seconds=30)
    iss_itrs = iss.pos_itrs(date)
    pos_from_ground = iss_itrs - eiffeltour_itrs
    angle = Maths.angle_vects(pos_from_ground, eiffeltour_itrs)
    if angle < visibility_angle: # ISS visible from Paris, Eiffel Tour
        print("ISS visible from Paris, Eiffel Tour, at max angle {0} radians, around {1}".format(visibility_angle, date))
        visible = True
        break

if not(visible):
    print("ISS not visible for a long time!")