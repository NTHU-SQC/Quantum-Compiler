# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:36:21 2021

@author: QEL
"""
import shape_function as sf
from Wavetools import Wave


def rabi_measure_waveform(
        sigma=100,
        flat_pi=80,
        flat_readout=2400,
        initial_delay=2000,
        end_deadtime=2000
        ):
    output = T1_measure_waveform(
            sigma, flat_pi, flat_readout, initial_delay, 0, end_deadtime)
    output.set_wire_names(['X', 'Readout'])
    output.set_name('Rabi')
    return output


def T1_measure_waveform(
        sigma=100,
        flat_pi=80,
        flat_readout=2400,
        initial_delay=2000,
        seperation=1000,
        end_deadtime=2000
        ):
    # pi pulse, param = [x, first_peak, flat, sigma]
    x_len_pi = 6 * sigma + flat_pi
    x_pi = sf.get_x(x_len_pi, x_len_pi + 1)
    param_pi = [x_pi, 3 * sigma, flat_pi, sigma]
    pi_pulse = Wave(sf.gaussian_square, param_pi) >> (initial_delay)
    # readout pulse, param = [x, first_peak, flat, sigma]
    x_len_readout = 6 * sigma + flat_readout
    x_readout = sf.get_x(x_len_readout, x_len_readout + 1)
    param_readout = [x_readout, 3 * sigma, flat_readout, sigma]
    read_pulse = Wave(sf.gaussian_square, param_readout) >> (
            initial_delay + x_len_pi + seperation) << (end_deadtime)
    output = ~~pi_pulse / ~~read_pulse
    output.set_wire_names(['X', 'Readout'])
    output.set_name('T1')
    return output


def T2_Ramsey_measure_waveform(
        sigma=100,
        flat_pi=80,
        flat_readout=2400,
        initial_delay=4000,
        seperation=1000,
        end_deadtime=1000
        ):
    # pi pulse, param = [x, first_peak, flat, sigma]
    x_len_pi = 6 * sigma + flat_pi
    x_pi = sf.get_x(x_len_pi, x_len_pi + 1)
    param_pi = [x_pi, 3 * sigma, flat_pi, sigma]
    pi_pulse = Wave(sf.gaussian_square, param_pi) >> (initial_delay)
    # null readout pulse
    null_readout = Wave(sf.const, [pi_pulse.get_x(), False])
    # obtain T2 QubitChannel
    T2_QubitChannel = ~~pi_pulse / ~~null_readout
    # rabi measure QubitChannel
    rabi_measure = T1_measure_waveform(
        sigma, flat_pi, flat_readout, seperation, 0, end_deadtime)
    output = T2_QubitChannel + rabi_measure
    output.set_wire_names(['X', 'Readout'])
    output.set_name('T2 Ramsey')
    return output


if __name__ == '__main__':
    b = T2_Ramsey_measure_waveform()
    b.plot(size=[10, 2.4])
