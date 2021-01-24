# EarthOrbit Python package

This package is aimed to track a satellite in orbit around Earth. It has been created in order to help small space centers (from universities for example) to do calculations for their cubesats. But you can use it in any way you want!

## Downloading
Using `pip`
```
pip install -i https://test.pypi.org/simple/ earthorbit
```

PyPI link: https://test.pypi.org/project/earthorbit/

## Depedencies
Using `numpy`, `arrow`, `requests`, `pyquaternion`

## How to use it?

First you need the **orbital information** of your satellite. This can either be:

* Classical orbital elements:
    * Eccentricity
    * Inclination
    * Right ascenscion of ascending node
    * Argument of pericenter
    * Mean motion
    * Mean anomaly
* State vectors:
    * Position (x, y, z)
    * Velocity (vx, vy, vz)
* A TLE
* Satellite NORAD cat ID

And, of course, an *epoch* at which the informations are relevant.

# Simple example: track ISS

Firstly we need to gather the orbital informations (aka. ground perturbations) of ISS. We can use Celestrak to do that.
But this library can do all the work for you: all you need to know is the NORAD catalog ID for the satellite (for ISS it is `25544`)
*If your satellite is not in the catalog, you can enter manually the orbital elements via classical elements, state vectors, TLE, etc. See documentation*

```python
import arrow
from earthorbit.orbit import Orbit

iss = Orbit.from_celestrak_norad_cat_id(25544)
print(iss.orbital_elements)
```

We created the orbit by gathering its latest orbital elements. We can now compute its position in diverse frames.
In order to track ISS, we are going to prompt GPS coordinates every seconds (for a minute). 
You can open an [online tracker](https://www.esa.int/Science_Exploration/Human_and_Robotic_Exploration/International_Space_Station/Where_is_the_International_Space_Station) to check if the position is accurate enough.
*Note:* there is about 1° of delay between an online tracker and our script

```python
for i in range(0, 60):
    date = arrow.utcnow()
    gps = iss.pos_gps(date)
    print(date.isoformat())
    print("ISS longitude: " + str(gps[1]) + "°")
    print("ISS latitude: " + str(gps[2]) + "°")
    print("----------------------")
    time.sleep(1)
```

# Another example: sunrise/sunset for ISS

We have to make a simulation of the orbit during a period of time:

```python
from earthorbit.simulation import Simulation

simulation_duration = 60*60*24 # [s]
simu = Simulation(iss, iss.epoch, simulation_duration, iss.name, [], [], print_progress=True) # empty list are for groundstations and ROI, which we do not need here
print("Simulation of ISS trajectory from {} to {}".format(simu.start_epoch, simu.stop_epoch))
sun_events = simu.sun_events # the events corresponding to Sun visibility
for s in sun_events:
    sunrise = arrow.get(s["start_unixepoch"])
    sunset = arrow.get(s["stop_unixepoch"])
    print("Daylight inside ISS from {} to {}!".format(sunrise, sunset))
```

# What's next?
- TLE propagator for more accurate predictions
- More warnings and better algorithms to detect problems
