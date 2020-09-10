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

req = requests.get("https://celestrak.com/NORAD/elements/gp.php?CATNR=25544&FORMAT=JSON")
print("Requested GP on Celestrak: {0}".format(req.text))
iss = Orbit.from_celestrak_json(req.text)
print("Orbit created with elements: {0}".format(iss.orbital_elements))

for i in range(0, 60):
    date = arrow.utcnow()
    gps = iss.pos_gps(date)
    print(date.isoformat())
    print("ISS longitude: {0}°".format(str(gps[1])))
    print("ISS latitude: {0}°".format(str(gps[2])))
    print("----------------------")
    time.sleep(1)