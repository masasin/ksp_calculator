#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

# Import frequently used functions and constants
from numpy import cos, sin, arccos, pi, sqrt
from pint import UnitRegistry

ureg = UnitRegistry()

m = ureg.m
km = ureg.km
Mm = ureg.Mm
Gm = ureg.Gm

kg = ureg.kg
Pa = ureg.Pa

ureg.define("month = 38.255581650418016 * hour = month")
ureg.define("year = 2556.5362979433785 * hour = year")

s = ureg.s
minute = ureg.minute
hour = ureg.hour
ureg.define("day = 6*hour = day")
day = ureg.day
month = ureg.month
year = ureg.year

deg = ureg.degrees
rad = ureg.rad

# Physical Constants
#G = 6.674e-11  # gravitational constant in KSP; m**3 / kg / s**2
G = 6.674e-11 * m**3 / kg / s**2  # gravitational constant in KSP; m**3 / kg / s**2
#ATM = 101325  # Pa at sea level
ATM = 101325 * Pa # Pa at sea level

# Conversion factors
#minute = 60  # seconds
#hour = 60 * minute
#day = 24 * hour
#year = 365.25 * day

#km = 1000  # metres
#Mm = 1000 * km
#Gm = 1000 * Mm


class Orbiter(object):
    def set_parent(self, parent):
        """
        Set the body around which we are orbiting
        """
        self.parent = parent
        if self not in self.parent.children:
            self.parent.children.append(self)

    def set_orbit_alt(self, a1, a2):
        """
        Set the orbit using the altitudes of the two apsides above the surface.

        These are the values shown in map view.
        """
        self.set_orbit(a1 + self.parent.radius,
                       a2 + self.parent.radius)

    def set_orbit(self, h1, h2):
        """
        Set the orbit using the heights of the two apsides above the centre.

        These are the values that really matter.
        """
        apo = max(h1, h2)
        peri = min(h1, h2)

        #self._mu = G * (self.mass + self.parent.mass)
        self._mu = G * self.mass

        self.apo = apo.to_base_units()  # m
        self.peri = peri.to_base_units()  # m
        self.SMA = (self.apo + self.peri) / 2  # Semi-major axis; m
        self.ecc = (self.apo - self.peri) / (self.apo + self.peri)
        self.sma = self.SMA * sqrt(1 - self.ecc**2)
        self.T_orb = 2 * pi * sqrt(self.SMA**3 / self._mu)  # siderial period

        # Synodic periods of heliocentric bodies are with Kerbin.
        # if self.parent is Kerbin.parent:
            # self.resonance, self.resonance = self.get_synod(Kerbin)
        # Synodic periods of moons and satellites are with parent bodies.
        # else:
            # self.resonance, self.resonance = self.get_synod(self.parent)

        # Calculate SOI of celestials
        if isinstance(self, Body):
            # Sphere of influence
            self.SOI = self.SMA * (self.mass / self.parent.mass)**(2/5)  # m

            # Determine if a synchronous orbit is possible
            if self.alt_sync > self.SOI or self.alt_sync < self.radius:
                self.sync_orbit_possible = False
            else:
                self.sync_orbit_possible = True

        self.spec_rel_ang_momentum = sqrt(self.SMA * (1 - self.ecc**2)
                                          * self._mu)
        self.spec_orb_energy = -self._mu / (2 * self.SMA)

    def set_incl(self, incl, arg_peri=0, long_AN=0):
        """
        Set the inclination of the orbit.

        All values are in degrees.
        """
        self.incl = incl.to(deg)  # deg
        self.arg_peri = arg_peri.to(deg)  # deg
        self.long_AN = long_AN.to(deg)   # deg
        self.long_peri = self.arg_peri + self.long_AN  # deg

    def set_pos_anomaly(self, mean_anomaly, epoch=0*rad):
        """
        Set the position anomaly of the body in the orbit.

        All values are in radians.
        """
        self.t_0 = epoch.to(rad)
        self.mean_anomaly_0 = mean_anomaly.to(rad)
        self.ecc_anomaly_0 = self.get_ecc_anomaly(self.mean_anomaly_0)
        self.true_anomaly_0 = arccos((cos(self.ecc_anomaly_0) - self.ecc) /
                                    (1 - self.ecc * cos(self.ecc_anomaly_0)))

    # TODO
    # Relative velocity. Not implemented.
    # 1 is moving away from planet. -1 is moving towards.
    # def set_pos_rv(self, radius, velocity, direction=1, epoch=0):
        # self.t_0 = epoch

    def get_synod(self, ref):
        """
        Return the synodic period and resonance with a reference body.
        """
        if self is not ref:
            T_synod = (ref.T_orb * self.T_orb) / abs(ref.T_orb - self.T_orb)
            resonance = T_synod / ref.T_orb
        else:
            T_synod = resonance = None
        return T_synod, resonance

    # TODO
    # Make more efficient solver
    def get_ecc_anomaly(self, mean_anomaly, precision=1e5):
        """
        Return the eccentric anomaly given the mean anomaly.

        The precision can be set.
        """
        self.mean_long_0 = self.mean_anomaly_0 + self.long_peri# * pi/180  # rad
        ecc_anomalies = np.linspace(0, 2*pi, precision, endpoint=True) * rad  # rad
        # use eccentric anomaly formula to get closest ecc_anomaly
        mean_anomalies = ecc_anomalies - self.ecc * sin(ecc_anomalies)
        return ecc_anomalies[np.abs(mean_anomalies - mean_anomaly).argmin()]

    def get_true_anomaly(self, mean_anomaly, precision=1e5):
        """
        Return the true anomaly given the mean anomaly.

        The precision can be set.
        """
        ecc_anomaly = self.get_ecc_anomaly(mean_anomaly, precision)
        return arccos((cos(ecc_anomaly) - self.ecc) /
                      (1 - self.ecc * cos(ecc_anomaly)))

    def get_mean_anomaly(self, dt):  # 0 to 2*pi
        """
        Return the mean anomaly at a given time.
        """
        mean_motion = sqrt(self._mu / self.SMA**3)  # by definition
        return (self.mean_anomaly_0 + mean_motion * dt).magnitude % (2*pi) * rad

    def get_h_time(self, time=0):
        """
        Return the height above the centre of the parent body at a given time.
        """
        true_anomaly = self.get_true_anomaly(self.get_mean_anomaly(time))
        return self.SMA * (1 - self.ecc ** 2) / \
            (1 + self.ecc * cos(true_anomaly))

    def get_speed_h(self, h):
        """
        Return the speed at a given distance from the focus.
        """
        return sqrt(self._mu * (2 / h - 1/self.SMA))

    def get_speed_time(self, time=0):
        """
        Return the speed at a given time.
        """
        h = self.get_h_time(time)
        return self.get_speed_h(h)

    def get_flight_ang_h(self, h):
        """
        Return the flight path angle at a given distance from the focus.
        """
        v = self.get_speed_h(h)
        return arccos(self.spec_rel_ang_momentum / (h * v))

    def get_flight_ang_time(self, time=0):
        """
        Return the flight path angle at a given time.
        """
        h = self.get_h_time(time)
        return self.get_flight_ang_h(h)

    def get_SMA_period(self, period):
        """
        Return the SMA required to have a given period.
        """
        return (period ** 2 / (4 * np.pi ** 2 / self._mu)) ** (1/3)

    def get_orbit_h_v(self, h, vel_vec):
        """
        Return the apsides given a height and a velocity vector.

        The velocity vector has polar coordinates:
            v in m/s
            theta in radians above the tangent to the surface
        """
        (v, theta) = vel_vec
        a = 1 / ((2 / h) - (v ** 2 / self._mu))
        momentum = h * v * cos(theta)
        return np.roots(1 / a, -2, momentum ** 2 / self._mu)

    def print_shape(self):
        """
        Print the shape of the orbit.
        """
        if isinstance(self, Body):
            print("Apoapsis: {.0f} m ({.0f} m ASL)".format
                  (self.apo, self.apo - self.radius))
            print("Periapsis: {.0f} m ({.0f} m ASL)".format
                  (self.peri, self.peri - self.radius))
        else:
            print("Apoapsis: {.0f} m".format(self.apo))
            print("Periapsis: {.0f} m".format(self.peri))
        print("Period: {.0f} s".format(self.T_orb))
        print("Inclination {.2f} deg".format(self.incl))


# A celestial is any large roundish object that has significant gravity.
class Celestial(object):
    def set_body(self, radius, mass, elev_max=0, has_atmosphere=False):
        """
        Create the celestial.
        """
        # radius
        self.radius = radius.to_base_units()  # m
        self.surface_area = 4 * pi * self.radius**2  # m**2
        self.volume = 4/3 * pi * self.radius**3  # m**3
        self.circumference = 2 * pi * self.radius  # m
        self.elev_max = elev_max.to_base_units()

        # mass
        self.mass = mass.to(kg)  # kg
        self.param_grav = G * self.mass  # m**3 / s**2

        # gravity
        self.density = self.mass / self.volume  # kg / m**3
        self.g_surf = self.get_g_alt(0 * m)  # N / kg
        self.v_esc = self.get_v_esc_alt(0 * m)  # m / s

        # atmosphere
        self.has_atmosphere = has_atmosphere

    def set_rot(self, T_rot, tilt_axis=0):
        """
        Define the celestial's rotation around its own axis.
        """
        self.tilt_axis = tilt_axis.to(deg)  # deg
        self.T_rot = T_rot.to_base_units()  # s
        self.v_rot = self.circumference / self.T_rot  # m / s
        self.w_rot = 2 * pi * rad / self.T_rot  # rad / s
        self.h_sync = (self.param_grav / self.w_rot**2) ** (1/3) * rad**(2/3) # m
        self.alt_sync = self.h_sync - self.radius  # m

    def set_atmosphere(self, pressure, scale_height, has_oxygen):
        """
        Create the celestial's atmosphere.
        """
        self.pressure_surf = pressure.to(Pa)  # Pa
        self.scale_height = scale_height.to_base_units()  # m
        self.has_oxygen = has_oxygen
        self.alt_atmo = -np.log(1e-6) * self.scale_height  # m

    def get_g_h(self, h):
        """
        Return the acceleration due to gravity at a given height.
        """
        return self.param_grav / h**2  # N / kg

    def get_g_alt(self, alt):
        """
        Return the acceleration due to gravity at a given altitude.
        """
        return self.get_g_h(self.radius + alt)  # N / kg

    def get_v_esc_h(self, h):
        """
        Return the velocity needed to escape the SOI at a given height.
        """
        return sqrt(2 * self.param_grav / h)  # m / s

    def get_v_esc_alt(self, alt):
        """
        Return the velocity needed to escape the SOI at a given altitude.
        """
        return self.get_v_esc_h(self.radius + alt)  # m / s

    def get_v_orb_h(self, h):  # m / s
        """
        Return the velocity needed to maintain an orbit with a given SMA.
        """
        return sqrt(self.param_grav / h)

    def get_v_orb_alt(self, alt):  # m / s
        """
        Return the velocity needed to maintain an orbit with a given altitude.

        The SMA is larger than the radius by this amount.
        """
        return sqrt(self.param_grav / (self.radius + alt))

    def get_t_leg(self, h1, h2):
        """
        Return transfer time between periapsis and apoapsis.
        """
        return np.pi * sqrt((h1 + h2)**3 / (8 * self.param_grav))

    def get_t_transfer_alt(self, a1, a2, at=None):
        """
        Return time for a full transfer between two altitudes.
        """
        if at is None:
            at = a2
        h1 = a1 + self.radius
        h2 = a2 + self.radius
        ht = at + self.radius
        self.get_t_transfer_h(h1, h2, ht)

    def get_t_transfer_h(self, h1, h2, ht=None):
        """
        Return time for a full transfer between two heights.
        """
        if ht is None:
            ht = h2
        return self.get_t_leg(h1, ht) + self.get_t_leg(ht, h2)

    def vis_viva(self, h1, h2, r):
        """
        Return the velocity at a given height.
        """
        a = (h1 + h2) / 2
        return sqrt(2 * self.param_grav / r - self.param_grav / a)

    def get_dv_transfer_alt(self, a1, a2, at=None):
        """
        Return delta-v budget of a transfer between two altitudes.
        """
        if at is None:
            at = a2
        h1 = a1 + self.radius
        h2 = a2 + self.radius
        ht = at + self.radius
        return self.get_dv_transfer_h(h1, h2, ht)

    def get_dv_transfer_h(self, h1, h2, ht=None):
        """
        Return delta-v budget of a transfer between heights.
        """
        if ht is None:
            ht = h2

        dv1 = self.vis_viva(ht, h1, h1) - self.vis_viva(h1, h1, h1)
        dv2 = self.vis_viva(ht, h2, ht) - self.vis_viva(ht, h1, ht)
        dv3 = self.vis_viva(h2, h2, h2) - self.vis_viva(ht, h2, h2)

        return dv1, dv2, dv3

    def get_phase_transfer_h(self, h1, h2):
        """
        Return the phase angle necessary for a Hohmann transfer.
        """
        t_inter = self.get_t_leg(h1, h2)
        w_target = sqrt(self.param_grav / h2) / h2
        angle_travelled = w_target * t_inter
        phase_angle = 180 - np.degrees(angle_travelled)
        if h1 > h2:
            return (phase_angle.magnitude % 180 - 180) * deg
        else:
            return phase_angle.magnitude % 180 * deg

    # TODO: Account for eccentricities and inclinations
    def get_transfer_params(self, s1, s2, a_park=100000):
        """
        Return transfer parameters.

        Input:
            Body or height

        Returns:
            Time of travel
            Phase Angle
            DV for the interplanetary leg
            Ejection velocity
            DV required for the outbound leg
            Ejection angle
        """
        if isinstance(s1, Body):
            assert s1.parent == self, "{} orbits a different body".format(s1)
            parent = s1
            h1 = parent.SMA
        else:
            h1 = s1
            parent = None

        if isinstance(s2, Body):
            assert s2.parent == self, "{} orbits a different body".format(s2)
            h2 = s2.SMA
        else:
            h2 = s2

        t_inter = self.get_t_leg(h1, h2)
        phase_angle = self.get_phase_transfer_h(h1, h2)
        dv_inter = self.get_dv_transfer_h(h1, h2)[0]

        if parent is not None:
            h_park = parent.radius + a_park
            v_esc = sqrt(dv_inter ** 2 + parent.param_grav *
                         (2 / h_park - 2/parent.SOI))
            dv_esc = v_esc - parent.vis_viva(h_park, h_park, h_park)

            #H = h_park * v_esc
            #epsilon = v_esc ** 2 / 2 - parent.param_grav / h_park
            #if v_esc > parent.get_v_esc_h(h_park):
                #e = sqrt(1 + 2 * epsilon * H**2 / parent.param_grav**2)
                #delta = np.arccos(1/e)
                #ejection_angle = 180 - np.degrees(delta)
            #else:
                #e = h_park * v_esc ** 2 / parent.param_grav - 1
                #a = h_park / (1 - e)
                #delta = np.arccos((a * (1 - e**2) - parent.SOI)
                                   #/ (e * parent.SOI))
                #ejection_angle = np.degrees(delta)

            e = h_park * v_esc ** 2 / parent.param_grav - 1
            a = h_park / (1 - e)
            delta = np.arccos((a * (1 - e**2) - parent.SOI) / (e * parent.SOI))
            ejection_angle = np.degrees(delta)

        else:
            v_esc = 0
            dv_esc = dv_inter
            ejection_angle = 0
        return t_inter, phase_angle, dv_inter, v_esc, dv_esc, ejection_angle

    def get_pressure_alt(self, alt):  # Pa
        """
        Return the atmospheric pressure at a given altitude.
        """
        return self.pressure_surf * np.exp(-alt/self.scale_height)

    def get_alt_pressure(self, pressure):  # m
        """
        Return the altitude at which a given atmospheric pressure occurs.
        """
        return self.scale_height * np.log(self.pressure_surf / pressure)

    def get_cycler(self, s1, s2, n_cycles=1, error=0.01, estimate=False):
        """
        Return the orbital parameters for a cycler between two bodies.

        Input:
            Bodies
            Number of full cycles

        Returns:
            Time per cycle
            SMA
            Apoapsis
            Argument of periapsis
            Ascending leg
            Descending leg
        """

        assert isinstance(s1, Body), "{} is not a body".format(s1)
        assert isinstance(s2, Body), "{} is not a body".format(s2)
        assert s1.parent == self, "{} orbits a different body".format(s1)
        assert s2.parent == self, "{} orbits a different body".format(s2)

        # Ensure that we are going from an inner to an outer body
        if s1.SMA > s2.SMA:
            s1, s2 = s2, s1

        resonance = (2 + 1/7) * ureg.dimensionless

        if estimate is False:
            resonance = s2.get_synod(s1)[1]
            modulo = np.modf(resonance)[0]
            n_round = 1 / modulo

        else:
            print("WARNING: Estimation mode. True cyclers may require course corrections.")
            res_real = s2.get_synod(s1)[1]
            mod_real, big_real  = np.modf(res_real)
            n_round = np.round(1 / mod_real)
            modulo = 1 / n_round
            resonance = big_real + modulo
            
        n = n_round * n_cycles
        n_periods = resonance * n_round
        cycle_length = n_periods * s1.T_orb

        T_orb_s2 = s1.T_orb * resonance / (resonance - 1)
        T_ratio = T_orb_s2 / s1.T_orb
        #print(resonance, s1.T_orb, T_orb_s2, T_orb_s2 - s1.T_orb, s2.T_orb)

        r = np.arange(1, n_periods - 1)
        peri = s1.SMA - s1.SOI * 1.01

        P = cycle_length * n / (n_round * r + n.magnitude % n_round.magnitude)
        a = (P/s1.T_orb) ** (2/3) * s1.SMA
        e = 1 - peri / a
        apo = a * (1 + e)
        p = a * (1 - e**2)
        omega = np.arccos((p - s1.SMA) / (s1.SMA * e))

        P_wrt_s1 = P / s1.T_orb
        T_s1 = np.array([])
        T_s2 = np.array([])

        for i in P_wrt_s1:
            b = i
            for multiplier in np.arange(n_periods) + 1:
                if abs(b - np.round(b)) > error:
                    b = i * multiplier
                else:
                    T_s1 = np.append(T_s1, b * s1.T_orb) * s
                    break

        for i in P_wrt_s1:
            b = i
            for multiplier in np.arange(n_periods) + 1:
                if abs(b / T_ratio - np.round(b / T_ratio)) > error:
                    b = i * multiplier
                else:
                    T_s2 = np.append(T_s2, b * s1.T_orb) * s
                    break

        print("{:d} options considered".format(int(n_periods.magnitude)))

        #print(P.size, a.size, apo.size, omega.size, T_s1.size, T_s2.size)
        try:
            result = np.column_stack([P, a, apo, omega, T_s1, T_s2])
        except:
            print("No solution found! Try increasing the allowable error.")
            return

        #result = result[result[:, 2] > s2.SMA]
        #print(result)

        if isinstance(self, Body):
            result = result[result[:, 2] < self.SOI]

        #if self.children[-1] is not s2:  # There are still bodies beyond that
            #sibling = self.children[self.children.index(s2) + 1]
            #if isinstance(sibling, Body):
                #result = result[result[:, 2] < sibling.peri - sibling.SOI]

        #final = result[np.argmin(result[:, -1] + result[:, -2])]
        arg = np.argmin(result[:, -1] + result[:, -2])
        
        print("Periapsis around {}: {:~}".format(self, peri))
        print("Apoapsis around {}: {:~}".format(self, apo[arg]))
        print("Orbit meets {} {:d}x and {} {:d}x in {:.0f} ({:.2f})"
              .format(s1, int((cycle_length / T_s1[arg]).magnitude),
                      s2, int((cycle_length / T_s2[arg]).magnitude),
                      cycle_length.to(day), cycle_length.to(year)))
        print("(Meeting is defined as closest approach ±{:.0f} degrees)"
              .format(error * 360))
        return P[arg], a[arg], apo[arg], omega[arg], T_s1[arg], T_s2[arg]
        #return final

    # When printing a body, show its name.
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class KerbalMade(object):
    def set_body(self, mass):
        # mass
        self.mass = mass.to(kg)  # kg
        self.param_grav = G * self.mass  # m**3 / s**2

    # When printing a body, show its name.
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class Star(Celestial):
    def __init__(self, name):
        self.name = name
        self.children = []
        self.is_orbiting = False


class Body(Celestial, Orbiter):
    # Planet or moon
    def __init__(self, name, parent):
        self.name = name
        self.children = []
        self.is_orbiting = True
        self.set_parent(parent)


class Spaceship(KerbalMade, Orbiter):
    def __init__(self, name, parent):
        self.name = name
        self.is_orbiting = True
        self.set_parent(parent)


def plot_params(b1, b2, fignum=None, accuracy=100):
    x = np.linspace(b1.parent.radius, b2.SMA, accuracy) / b1.SMA
    params = np.array([b1.parent.get_transfer_params(b1, b1.SMA * a)
                      for a in x])
    param_names = ("Time of flight", "Phase angle", "dv of transfer",
                   "Ejection velocity", "Ejection dv", "Ejection angle")
    if fignum is None:
        for i in range(params.shape[1]):
            plt.figure()
            plt.xlabel("{} SMAs from {}".format(b1, b1.parent))
            plt.ylabel(param_names[i])
            plt.title(param_names[i])
            plt.plot(x, params[:, i])
            for child in b1.parent.children:
                if isinstance(child, Body):
                    plt.text(child.SMA / b1.SMA, child.SMA / b1.SMA*20, child)
                    plt.axvspan((child.SMA - child.SOI)/b1.SMA, (child.SMA +
                                child.SOI)/b1.SMA, facecolor="g", alpha=0.5)
        plt.show()
    else:
        i = fignum
        plt.figure()
        plt.xlabel("{} SMAs from {}".format(b1, b1.parent))
        plt.ylabel(param_names[i])
        plt.title(param_names[i])
        plt.plot(x, params[:, i])
        for child in b1.parent.children:
            plt.text(child.SMA / b1.SMA, child.SMA / b1.SMA * 20, child)
            plt.axvspan((child.SMA - child.SOI)/b1.SMA,
                        (child.SMA + child.SOI)/b1.SMA,
                        facecolor="g", alpha=0.5)
        plt.show()


if __name__ == "__main__":
    Kerbol = Star("Kerbol")
    Moho = Body("Moho", Kerbol)
    Eve = Body("Eve", Kerbol)
    Gilly = Body("Gilly", Eve)
    Kerbin = Body("Kerbin", Kerbol)
    Mun = Body("Mun", Kerbin)
    Minmus = Body("Minmus", Kerbin)
    Duna = Body("Duna", Kerbol)
    Ike = Body("Ike", Duna)
    Dres = Body("Dres", Kerbol)
    Jool = Body("Jool", Kerbol)
    Laythe = Body("Laythe", Jool)
    Vall = Body("Vall", Jool)
    Tylo = Body("Tylo", Jool)
    Bop = Body("Bop", Jool)
    Pol = Body("Pol", Jool)
    Eeloo = Body("Eeloo", Kerbol)

    celestials_data = [
        {"id": 0, "body": Kerbol,
            "radius": 261600000 * m, "mass":  1.7565670e28 * kg, "max elevation": 0 * m,
            "T_rot": 432000 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 4, "body": Moho,
            "radius": 250000 * m, "mass":  2.5263617e21 * kg, "max elevation": 6752.3 * m,
            "T_rot": 1210000 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 5, "body": Eve,
            "radius": 700000 * m, "mass": 1.2244127e23 * kg, "max elevation": 6450 * m,
            "T_rot": 80500 * s, "axial tilt": 0 * deg, "has atmosphere?": True},
        {"id": 13, "body": Gilly,
            "radius": 13000 * m, "mass": 1.2420512e17 * kg, "max elevation": 6399.7 * m,
            "T_rot": 28255 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 1, "body": Kerbin,
            "radius": 600000 * m, "mass": 5.2915793e22 * kg, "max elevation": 4033.5 * m,
            "T_rot": 21600 * s, "axial tilt": 0 * deg, "has atmosphere?": True},
        {"id": 2, "body": Mun,
            "radius": 200000 * m, "mass": 9.7600236e20 * kg, "max elevation": 3334.7 * m,
            "T_rot": 138984.38 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 3, "body": Minmus,
            "radius": 60000 * m, "mass": 2.6457897e19 * kg, "max elevation": 5724.6 * m,
            "T_rot": 40400 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 6, "body": Duna,
            "radius": 320000 * m, "mass": 4.5154812e21 * kg, "max elevation": 4661.9 * m,
            "T_rot": 65517.859 * s, "axial tilt": 0 * deg, "has atmosphere?": True},
        {"id": 7, "body": Ike,
            "radius": 130000 * m, "mass": 2.7821949e20 * kg, "max elevation": 12724.9 * m,
            "T_rot": 65517.862 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 15, "body": Dres,
            "radius": 138000 * m, "mass": 3.2191322e20 * kg, "max elevation": 5669.6 * m,
            "T_rot": 34800 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 8, "body": Jool,
            "radius": 6000000 * m, "mass": 4.2332635e24 * kg, "max elevation": 0 * m,
            "T_rot": 36000 * s, "axial tilt": 0 * deg, "has atmosphere?": True},
        {"id": 9, "body": Laythe,
            "radius": 500000 * m, "mass": 2.9397663e22 * kg, "max elevation": 6008.9 * m,
            "T_rot": 52980.879 * s, "axial tilt": 0 * deg, "has atmosphere?": True},
        {"id": 10, "body": Vall,
            "radius": 300000 * m, "mass": 3.1088028e21 * kg, "max elevation": 7975.4 * m,
            "T_rot": 105962.09 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 12, "body": Tylo,
            "radius": 600000 * m, "mass": 4.2332635e22 * kg, "max elevation": 12694 * m,
            "T_rot": 211926.36 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 11, "body": Bop,
            "radius": 65000 * m, "mass": 3.7261536e19 * kg, "max elevation": 21748 * m,
            "T_rot": 544507.4 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 14, "body": Pol,
            "radius": 44000 * m, "mass": 1.0813636e19 * kg, "max elevation": 5584.8 * m,
            "T_rot": 901902.62 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        {"id": 16, "body": Eeloo,
            "radius": 210000 * m, "mass": 1.1149358e21 * kg, "max elevation": 3868.6 * m,
            "T_rot": 19460 * s, "axial tilt": 0 * deg, "has atmosphere?": False},
        ]

    for i in celestials_data:
        i["body"].set_body(i["radius"], i["mass"], i["max elevation"],
                           i["has atmosphere?"])
        i["body"].set_rot(i["T_rot"], i["axial tilt"])

    atmospheres_data = [
        {"body": Eve, "pressure": 506625 * Pa, "scale height": 7000 * m, "O2": False},
        {"body": Kerbin, "pressure": 101325 * Pa, "scale height": 5000 * m, "O2": True},
        {"body": Duna, "pressure": 20265 * Pa, "scale height": 3000 * m, "O2": False},
        {"body": Jool, "pressure": 1519880 * Pa, "scale height": 1e4 * m, "O2": False},
        {"body": Laythe, "pressure": 81060 * Pa, "scale height": 4000 * m, "O2": True},
    ]

    for i in atmospheres_data:
        i["body"].set_atmosphere(i["pressure"], i["scale height"], i["O2"])

    # Kerbin placed first for synodal orbit calculations
    orbits_init_data = [
        {"body": Kerbin, "apoapsis": 13599840256 * m, "periapsis": 13599840256 * m,
            "inclination": 0 * deg, "arg of peri": 0 * deg, "longitude of AN": 0 * deg,
            "positional anomaly at epoch": 3.14 * rad},
        {"body": Mun, "apoapsis": 12000000 * m, "periapsis": 12000000 * m,
            "inclination": 0 * deg, "arg of peri": 0 * deg, "longitude of AN": 0 * deg,
            "positional anomaly at epoch": 0.9 * rad},
        {"body": Minmus, "apoapsis": 47000000 * m, "periapsis": 47000000 * m,
            "inclination": 6 * deg, "arg of peri": 38 * deg, "longitude of AN": 78 * deg,
            "positional anomaly at epoch": 1.7 * rad},
        {"body": Moho, "apoapsis": 6315765980 * m, "periapsis": 4210510628 * m,
            "inclination": 7 * deg, "arg of peri": 15 * deg, "longitude of AN": 70 * deg,
            "positional anomaly at epoch": 3.14 * rad},
        {"body": Eve, "apoapsis": 9931011387 * m, "periapsis": 9734357701 * m,
            "inclination": 2.1 * deg, "arg of peri": 0 * deg, "longitude of AN": 15 * deg,
            "positional anomaly at epoch": 3.14 * rad},
        {"body": Gilly, "apoapsis": 48825000 * m, "periapsis": 14175000 * m,
            "inclination": 12 * deg, "arg of peri": 10 * deg, "longitude of AN": 80 * deg,
            "positional anomaly at epoch": 0.9 * rad},
        {"body": Duna, "apoapsis": 21783189163 * m, "periapsis": 19669121365 * m,
            "inclination": 0.06 * deg, "arg of peri": 0 * deg, "longitude of AN": 135.5 * deg,
            "positional anomaly at epoch": 3.14 * rad},
        {"body": Ike, "apoapsis": 3296000 * m, "periapsis": 3104000 * m,
            "inclination": 0.2 * deg, "arg of peri": 0 * deg, "longitude of AN": 0 * deg,
            "positional anomaly at epoch": 1.7 * rad},
        {"body": Dres, "apoapsis": 46761053522 * m, "periapsis": 34917642884 * m,
            "inclination": 5 * deg, "arg of peri": 90 * deg, "longitude of AN": 280 * deg,
            "positional anomaly at epoch": 3.14 * rad},
        {"body": Jool, "apoapsis": 72212238387 * m, "periapsis": 65334882253 * m,
            "inclination": 1.304 * deg, "arg of peri": 0 * deg, "longitude of AN": 52 * deg,
            "positional anomaly at epoch": 0.1 * rad},
        {"body": Laythe, "apoapsis": 27184000 * m, "periapsis": 27184000 * m,
            "inclination": 0 * deg, "arg of peri": 0 * deg, "longitude of AN": 0 * deg,
            "positional anomaly at epoch": 3.14 * rad},
        {"body": Vall, "apoapsis": 43152000 * m, "periapsis": 43152000 * m,
            "inclination": 0 * deg, "arg of peri": 0 * deg, "longitude of AN": 0 * deg,
            "positional anomaly at epoch": 0.9 * rad},
        {"body": Tylo, "apoapsis": 68500000 * m, "periapsis": 68500000 * m,
            "inclination": 0.025 * deg, "arg of peri": 0 * deg, "longitude of AN": 0 * deg,
            "positional anomaly at epoch": 3.14 * rad},
        {"body": Bop, "apoapsis": 158697500 * m, "periapsis": 98302500 * m,
            "inclination": 15 * deg, "arg of peri": 25 * deg, "longitude of AN": 10 * deg,
            "positional anomaly at epoch": 0.9 * rad},
        {"body": Pol, "apoapsis": 210624206 * m, "periapsis": 149155794 * m,
            "inclination": 4.25 * deg, "arg of peri": 15 * deg, "longitude of AN": 2 * deg,
            "positional anomaly at epoch": 0.9 * rad},
        {"body": Eeloo, "apoapsis": 113549713200 * m, "periapsis": 66687926800 * m,
            "inclination": 6.15 * deg, "arg of peri": 260 * deg, "longitude of AN": 50 * deg,
            "positional anomaly at epoch": 3.14 * rad}
        ]

    for i in orbits_init_data:
        i["body"].set_orbit(i["apoapsis"], i["periapsis"])
        i["body"].set_incl(i["inclination"], i["arg of peri"],
                           i["longitude of AN"])
        i["body"].set_pos_anomaly(i["positional anomaly at epoch"])

    # Lists
    planets = Kerbol.children
    moons = []
    for planet in planets:
        moons.extend(planet.children)

    #plot_params(Kerbin, Eeloo, accuracy=1000)
