"""
    track-iss.py

    This script is an example on how to use the module.
    It requests the GP of ISS via Celestrak, creates an orbit out of it, and gets the current GPS position of the ISS.
"""

from earthorbit.orbit import Orbit
import numpy as np
import arrow
import time
import requests

iss = Orbit.from_celestrak_norad_cat_id(25544) 

for i in range(0, 60):
    date = arrow.utcnow()
    gps = iss.pos_gps(date)
    print(date.isoformat())
    print("ISS longitude: {0}°".format(str(gps[1])))
    print("ISS latitude: {0}°".format(str(gps[2])))
    print("----------------------")
    time.sleep(1)