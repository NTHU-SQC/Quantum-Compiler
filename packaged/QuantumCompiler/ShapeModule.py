# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 18:09:39 2021

@author: Alaster

Create the following functions to build waveform
"""

import numpy as np
from inspect import getfullargspec as showarg


def gaussian(x: np.array, peak_x: float, sigma: float):
    """
    Gaussian pulse generating function

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    peak_x : float
        Position of Gaussian pulse peak.
    sigma : float
        Standard deviation.

    Returns
    -------
    np.array
        Output wave values.

    """
    return np.exp((-1 * (x - peak_x)**2) / (2 * sigma**2))


def const(x: np.array, lv: float):
    """
    Constant level.

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    lv : float
        Amplitude of wave.

    Returns
    -------
    np.array
        Output wave values.

    """
    return lv * np.ones(x.size)


def exp_rising(x: np.array, peak_x: float, tau: float):
    """
    Unit exponential rising pulse.

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    peak_x : float
        Position of pulse peak.
    tau : float
        Characteristic time for exponential pulse.

    Returns
    -------
    np.array
        Output wave values.

    """
    return np.concatenate((
        np.exp((x[x <= peak_x] - peak_x) / tau),
        np.zeros(x[x > peak_x].size)
        ))


def exp_falling(x: np.array, peak_x: float, tau: float):
    """
    Unit exponential falling pulse.

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    peak_x : float
        Position of pulse peak.
    tau : float
        Characteristic time for exponential pulse.

    Returns
    -------
    np.array
        Output wave values.

    """
    return np.concatenate((
        np.zeros(x[x < peak_x].size),
        np.exp((peak_x - x[x >= peak_x]) / tau)
        ))


def gaussian_square(
        x: np.array, first_peak_x: float, flat: float, sigma: float
        ):
    """
    Unit square pulse with Gaussian edges.

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    first_peak_x : float
        Position of the first Gaussian edge peak.
    flat : float
        Time span of flat top (1s).
    sigma : float
        Standard deviation of Gaussian edge.

    Returns
    -------
    np.array
        Output wave values.

    """
    return np.concatenate((
        gaussian(x[x <= first_peak_x], first_peak_x, sigma),
        np.ones(
            x[np.logical_and(x > first_peak_x, x <= first_peak_x + flat)].size
            ),
        gaussian(
            x[x > first_peak_x + flat], x[x > first_peak_x + flat][0], sigma
            )
        ))


def exp_square(x: np.array, first_peak_x: float, flat: float, tau: float):
    """
    Unit square pulse with exponential edges.

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    first_peak_x : float
        Position of the first Gaussian edge peak.
    flat : float
        Time span of flat top (1s).
    tau : float
        Characteristic time of exponential edge.

    Returns
    -------
    np.array
        Output wave values.

    """
    return np.concatenate((
        exp_rising(x[x <= first_peak_x], first_peak_x, tau),
        np.ones(
            x[np.logical_and(x > first_peak_x, x <= first_peak_x + flat)].size
            ),
        exp_falling(
            x[x > first_peak_x + flat], x[x > first_peak_x + flat][0], tau
            )
        ))


def sine(x: np.array, period: float, start_phase: float):
    """
    Unit sine pulse generating function specified in period.

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    period : float
        Wave period.
    start_phase : float
        Starting phase in radian.

    Returns
    -------
    np.array
        Output wave values.

    """
    return np.sin(2*np.pi*(x / period) + start_phase)


def sine2(x: np.array, frequency: float, start_phase: float):
    """
    Unit sine pulse generating function specified in frequency.

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    frequency : float
        Wave frequency.
    start_phase : float
        Starting phase in radian.

    Returns
    -------
    np.array
        Output wave values.

    """
    return np.sin(2 * np.pi * frequency * x + start_phase)


def cosine(x: np.array, period: float, start_phase: float):
    """
    Unit cosine pulse generating function specified in period.

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    period : float
        Wave period.
    start_phase : float
        Starting phase in radian.

    Returns
    -------
    np.array
        Output wave values.

    """
    return np.cos(2*np.pi*(x / period) + start_phase)


def cosine2(x: np.array, frequency: float, start_phase: float):
    """
    Unit cosine pulse generating function specified in frequency.

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    frequency : float
        Wave frequency.
    start_phase : float
        Starting phase in radian.

    Returns
    -------
    np.array
        Output wave values.

    """
    return np.cos(2 * np.pi * frequency * x + start_phase)


def square(x: np.array, start: float, flat: float):
    """
    Unit square pulse generating function.

    Parameters
    ----------
    x : np.array
        Wave event timeline, start from 0 is demanded.
    start : float
        Start time for 1s.
    flat : float
        Time span for 1s.

    Returns
    -------
    np.array
        Output wave values.

    """
    return np.concatenate((
        np.zeros(x[x < start].size),
        np.ones(x[np.logical_and(x >= start, x <= start + flat)].size),
        np.zeros(x[x > start + flat].size)
        ))


def get_x(span: float = .0, sampling_rate: float = 1e9):
    """
    Formatted timeline creation.

    Parameters
    ----------
    span : float, optional
        Overall length of timeline. The default is .0.
    sampling_rate : float, optional
        Sampling rate for DAC. The default is 1e9 (Suggested).

    Returns
    -------
    np.array

    """
    points = int(round(span * sampling_rate, 3)) + 1
    return np.linspace(0, points - 1, points) / sampling_rate


function_mappings = {
    'gaussian': gaussian,
    'const': const,
    'exp_rising': exp_rising,
    'exp_falling': exp_falling,
    'gaussian_square': gaussian_square,
    'exp_square': exp_square,
    'sine': sine,
    'sine2': sine2,
    'cosine': cosine,
    'cosine2': cosine2,
    'square': square,
    }


def setFunc(
        func: str, funcArg: list,
        span: float = .0, sampling_rate: float = 1e9,
        name: str = '', appendRule: list = [True, True]
        ):
    """
    Shape function wrapper.

    Parameters
    ----------
    func : str/function handle
        Shape function to be used, can be specified by string or function
        handle.
    funcArg : list/dict
        List/dict of arguments of the function. In dict mode, the user can
        declare key-value pair with each key corresponding to the argument of
        the function. The key must match the argument name.
    span : float, optional
        Overall length of timeline. The default is .0.
    sampling_rate : float, optional
        Sampling rate for DAC. The default is 1e9 (Suggested).
    name : string, optional
        Name of the wave object. The default is ''.
    appendRule : list, optional
        Wave concatenation rule. The default is [True, True].

    """
    if isinstance(func, str):
        func = function_mappings[func]
    argNames = showarg(func).args[1:]
    if isinstance(funcArg, dict):
        funcArg = [funcArg[arg] for arg in argNames]
    generator = {
        'function': func.__name__,
        'X': {'span': span, 'sampling_rate': sampling_rate},
        'Y': dict(zip(argNames, funcArg)),
        'name': name,
        'appendRule': appendRule
        }
    return generator


def parse(generator: dict):
    """
    Parse and compile generator data into wave x-y data

    Parameters
    ----------
    generator : dict
        Discription dict for wave generation.

    Returns
    -------
    x : np.array
        x (time) data.
    y : np.array
        y (amplitude) data.
    name : str
        Name of wave.
    appendRule : list
        Concatenation rule for waves.

    """
    # x
    span = generator['X']['span']
    sampling_rate = generator['X']['sampling_rate']
    x = get_x(span, sampling_rate)
    # y
    func = generator['function']
    if isinstance(func, str):
        func = function_mappings[func]
    argNames = showarg(func).args[1:]
    funcArg = [generator['Y'][arg] for arg in argNames]
    y = func(x, *funcArg)
    # name
    name = generator['name']
    # appendRule
    appendRule = generator['appendRule']
    return x, y, name, appendRule
