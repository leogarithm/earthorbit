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