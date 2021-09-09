# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 18:09:39 2021

@author: Alaster

Create the following functions to build waveform
"""

import numpy as np


def get_x(span=.0, points=0):
    """
    Formatted timeline creation.

    Parameters
    ----------
    span : float, optional
        Overall length of timeline. The default is .0.
    points : float, optional
        Number of points. The default is 0.

    Returns
    -------
    numpy.array

    """
    return np.linspace(0, span, points)


def gaussian(x, peak_x=.0, sigma=1.0, name=''):
    """
    Gaussian pulse generating function

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    peak_x : float, optional
        Position of Gaussian pulse peak. The default is .0.
    sigma : float, optional
        Standard deviation. The default is 1.0.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    x = x.astype(np.float)
    variables = {'function': gaussian, 'peak_x': peak_x, 'sigma': sigma}
    y = np.exp((-1 * (x - peak_x)**2) / (2 * sigma**2))
    return packer(x, y, variables, [False, False], name)


def const(x, setOne=True, name=''):
    """
    Constant level generating function

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    setOne : boolean, optional
        Toggle True(False) to output 1s(0s). The default is True.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    x = x.astype(np.float)
    variables = {'function': const, 'setOne': setOne}
    offset = 0
    if setOne:
        offset += 1
    y = np.zeros(x.size) + offset
    return packer(x, y, variables, [False, False], name)


def exp_rising(x, peak_x=0.0, tau=1.0, name=''):
    """
    Exponential rising pulse generating function

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    peak_x : float, optional
        Position of pulse peak. The default is 0.0.
    tau : float, optional
        Characteristic time for exponential pulse. The default is 1.0.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    x = x.astype(np.float)
    variables = {'function': exp_rising, 'peak_x': peak_x, 'tau': tau}
    y = np.concatenate((
        np.exp((x[x <= peak_x] - peak_x) / tau), np.zeros(x[x > peak_x].size)
        ))
    return packer(x, y, variables, [False, False], name)


def exp_falling(x, peak_x=0.0, tau=1.0, name=''):
    """
    Exponential falling pulse generating function

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    peak_x : float, optional
        Position of pulse peak. The default is 0.0.
    tau : float, optional
        Characteristic time for exponential pulse. The default is 1.0.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    x = x.astype(np.float)
    variables = {'function': exp_falling, 'peak_x': peak_x, 'tau': tau}
    y = np.concatenate((
        np.zeros(x[x < peak_x].size), np.exp((peak_x - x[x >= peak_x]) / tau)
        ))
    return packer(x, y, variables, [False, False], name)


def gaussian_square(x, first_peak_x=5.0, flat=10.0, sigma=1.0, name=''):
    """
    Gaussian square pulse generating function

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    first_peak_x : float, optional
        Position of the first Gaussian edge peak. The default is 5.0.
    flat : float, optional
        Time span of flat top (1s). The default is 10.0.
    sigma : float, optional
        Standard deviation of Gaussian edge. The default is 1.0.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    x = x.astype(np.float)
    variables = {'function': gaussian_square, 'first_peak_x': first_peak_x,
                 'flat': flat, 'sigma': sigma}
    y = np.concatenate((
        gaussian(x[x <= first_peak_x], first_peak_x, sigma)['y'],
        np.ones(
            x[np.logical_and(x > first_peak_x, x <= first_peak_x + flat)].size
            ),
        gaussian(
            x[x > first_peak_x + flat], x[x > first_peak_x + flat][0], sigma
            )['y']
        ))
    return packer(x, y, variables, [False, False], name)


def exp_square(x, first_peak_x=5.0, flat=10.0, tau=1.0, name=''):
    """
    Generating function for square pulse with exponential edges.

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    first_peak_x : float, optional
        Position of the first Gaussian edge peak. The default is 5.0.
    flat : float, optional
        Time span of flat top (1s). The default is 10.0.
    tau : float, optional
        Characteristic time of exponential edge. The default is 1.0.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    x = x.astype(np.float)
    variables = {'function': gaussian_square, 'first_peak_x': first_peak_x,
                 'flat': flat, 'tau': tau}
    y = np.concatenate((
        exp_rising(x[x <= first_peak_x], first_peak_x, tau)['y'],
        np.ones(
            x[np.logical_and(x > first_peak_x, x <= first_peak_x + flat)].size
            ),
        exp_falling(
            x[x > first_peak_x + flat], x[x > first_peak_x + flat][0], tau
            )['y']
        ))
    return packer(x, y, variables, [False, False], name)


def sine(x, period=10.0, start_phase=0.0, name=''):
    """
    Sine pulse generating function

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    period : float, optional
        Wave period. The default is 10.0.
    start_phase : float, optional
        Starting phase in radian. The default is 0.0.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    x = x.astype(np.float)
    variables = {
            'function': sine, 'period': period, 'start_phase': start_phase}
    y = np.sin(2*np.pi*(x / period) + start_phase)
    return packer(x, y, variables, [True, True], name)


def sine2(x, frequency=10.0, start_phase=0.0, name=''):
    """
    Sine pulse generating function

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    frequency : float, optional
        Wave frequency. The default is 10.0.
    start_phase : float, optional
        Starting phase in radian. The default is 0.0.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    x = x.astype(np.float)
    variables = {
            'function': sine,
            'frequency': frequency,
            'start_phase': start_phase
            }
    y = np.sin(2 * np.pi * frequency * x + start_phase)
    return packer(x, y, variables, [True, True], name)


def cosine(x, period=10.0, start_phase=0.0, name=''):
    """
    Cosine pulse generating function

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    period : float, optional
        Wave period. The default is 10.0.
    start_phase : float, optional
        Starting phase in radian. The default is 0.0.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    return sine(x, period, start_phase + np.pi/2, name)


def cosine2(x, frequency=10.0, start_phase=0.0, name=''):
    """
    Cosine pulse generating function

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    frequency : float, optional
        Wave frequency. The default is 10.0.
    start_phase : float, optional
        Starting phase in radian. The default is 0.0.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    return sine2(x, frequency, start_phase + np.pi/2, name)


def square(x, start=.0, flat=1.0, name=''):
    """
    Square pulse generating function

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    start : float, optional
        Start time for 1s. The default is .0.
    flat : float, optional
        Time span for 1s. The default is 1.0.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    x = x.astype(np.float)
    variables = {'function': square, 'start': start, 'flat': flat}
    y = np.concatenate((
        np.zeros(x[x < start].size),
        np.ones(x[np.logical_and(x >= start, x <= start + flat)].size),
        np.zeros(x[x > start + flat].size)
        ))
    return packer(x, y, variables, [False, False], name)


def packer(x, y, variables={}, appendRule=[False, False], name=''):
    """
    Given x, y ,create a corresponding dictionary package.

    Parameters
    ----------
    x : numpy.array
        Wave event timeline, start from 0 is demanded.
    y : numpy.array
        Amplitude data array.
    variables : dict, optional
        Some information about the wave.
    name : string, optional
        Name of the wave object. The default is ''.

    Returns
    -------
    properties : dict
        A dictionary of relavent data to be encapsulated.

    """
    properties = {
            'name': name,
            'variables': variables,
            'x': x,
            'y': y,
            'appendRule': appendRule
            }
    return properties


if __name__ == '__main__':
    pass
