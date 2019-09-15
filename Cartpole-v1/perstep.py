import math
import numpy as np


FORCE_MAG = 10.0
GRAVITY = 9.8
MASSPOLE = 0.1
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH
MASSCART = 1.0
TOTAL_MASS = MASSPOLE + MASSCART
KINEMATICS_INTEGRATOR = 'euler'
TAU = 0.02
x_threshold = 2.4
theta_threshold_radians = 12 * 2 * math.pi / 360
STEPS_BEYOND_DONE = None


def virtual_step(state, action):
    global STEPS_BEYOND_DONE
    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) /\
        TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (
            4.0 / 3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS))
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    if KINEMATICS_INTEGRATOR == 'euler':
        x = x + TAU * x_dot
        x_dot = x_dot + TAU * xacc
        theta = theta + TAU * theta_dot
        theta_dot = theta_dot + TAU * thetaacc
    else:  # semi-implicit euler
        x_dot = x_dot + TAU * xacc
        x = x + TAU * x_dot
        theta_dot = theta_dot + TAU * thetaacc
        theta = theta + TAU * theta_dot

    state = (x, x_dot, theta, theta_dot)
    done = x < -x_threshold \
           or x > x_threshold \
           or theta < -theta_threshold_radians \
           or theta > theta_threshold_radians
    done = bool(done)

    if not done:
        reward = 1.0
    elif STEPS_BEYOND_DONE is None:
        # Pole just fell!
        STEPS_BEYOND_DONE = 0
        reward = 1.0
    else:
        STEPS_BEYOND_DONE += 1
        reward = 0.0

    return np.array(state), reward, done, {}



