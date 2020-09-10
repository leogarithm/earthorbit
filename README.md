# EarthOrbit Python package

This package is aimed to track a satellite in orbit around Earth. It has been created in order to help small space centers (from universities for example) to do calculations for their cubesats. But you can use it in any way you want!

## Depedencies
Using `numpy` and `arrow`

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

And, of course, an *epoch* at which the informations are relevant.

If your satellite is already in orbit, you can retrieve the *classical orbital elements* in *.json* format using [Celestrak.com](https://celestrak.com), which we assume you are familiar with, if you have a satellite in orbit.



## What about units?

This library uses **SI units.** Therefore if you have to work with non-SI units, you have to do the conversions yourself.
For example, *distances* are in *meters*, *velocities* in *meters per second*, *angles* are in *radians*, and so on.

## What will be added in next releases?

* QSW satellite frame conversions
* Celestial coordinates conversion (azimuth, elevation)
* More stuff, probably...

# Example: track ISS

Firstly we need to gather the orbital informations (aka. ground perturbations) of ISS. We can use Celestrak to do that.
Open https://celestrak.com/NORAD/elements/gp.php?CATNR=25544&FORMAT=JSON, copy the raw text and stores it in a text variable.
*Note:* you can retrieve the informations of any satellite you want using Celestrak, just follow [the tutorial](https://celestrak.com/NORAD/documentation/gp-data-formats.php).
You can also use `requests` Python library to make a HTML GET request to the Celestrak page.

```python
    from earthorbit.orbit import Orbit
    import arrow
    import time

    orb_els = '{"OBJECT_NAME":"ISS (ZARYA)","OBJECT_ID":"1998-067A","EPOCH":"2020-08-30T05:35:34.243872","MEAN_MOTION":15.49199437,"ECCENTRICITY":0.000186,"INCLINATION":51.647,"RA_OF_ASC_NODE":342.0279,"ARG_OF_PERICENTER":69.3228,"MEAN_ANOMALY":129.2061,"EPHEMERIS_TYPE":0,"CLASSIFICATION_TYPE":"U","NORAD_CAT_ID":25544,"ELEMENT_SET_NO":999,"REV_AT_EPOCH":24349,"BSTAR":3.9609e-5,"MEAN_MOTION_DOT":1.748e-5,"MEAN_MOTION_DDOT":0}'
```

*Note:* in this example we use a ground perturbations that I searched on 30th of August, 2020, as you can see in the string. **If you use this exact string instead of making a new request to Celestrak and use the most recents orbital elements, the results will probably not be accurate.**

Now we create the corresponding orbit, and print the orbital elements to see if it is accurate to Celestrak

```python
    iss = Orbit.from_celestrak_json(celestrak_gp)
    print(iss.orbital_elements)
```

And now we are creating a loop that will prompt the GPS position of the ISS every seconds (for a minute). You can open an [online tracker](http://www.isstracker.com/) to check if the position is accurate enough.
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