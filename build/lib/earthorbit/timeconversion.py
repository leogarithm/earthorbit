import math

class TimeConversion:
    @staticmethod
    def unixepoch2jd(unixepoch: int) -> float:
        """
        Compute the Julian day (JD) of the given unixepoch (POSIX time)

        :param unixepoch: Unixepoch (POSIX time). units: seconds [s]
        :return: Julian day (JD). units: JD 
        """
        return unixepoch/86400 + 2440587.5

    @staticmethod
    def unixepoch2j2000(unixepoch: int) -> float:
        """
        Compute the Julian year (J2000) of the given unixepoch (POSIX time)

        :param unixepoch: Unixepoch (POSIX time). units: seconds [s]
        :return: Julian year (J2000). units: J2000
        """
        return unixepoch/86400 - 10957.5

    @staticmethod
    def unixepoch2daysfrommillennium(unixepoch: int) -> float:
        """
        Compute date in fractions of days since 1 january 2000 00:00, of the given unixepoch (POSIX time)
        Taken from: https://stjarnhimlen.se/comp/ppcomp.html#3

        :param unixepoch: Unixepoch (POSIX time). units: seconds [s]
        :return: Date (fraction of days) since 1 january 2000 00:00
        """
        return TimeConversion.unixepoch2j2000(unixepoch) + 2.5
    
    @staticmethod
    def revperday2radpersec(revperday: float) -> float:
        """
            Converts a quantity in [revolution/day] to [rad/s]

            :param revperday: quantity in [revolution/day] to be converted
            :type revperday: float
            :return: quantity converted in [rad/s]
            :rtype: float
        """
        return revperday*7.27220521664304e-05


    