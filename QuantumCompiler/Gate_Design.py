# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 20:40:29 2021
"""
from .ShapeModule import setFunc
from . import TemplateModule as tpm
from .WaveModule import Wave, Waveform
import numpy as np
from .QuantumCircuit import QuantumCircuit
import random

def dispose_point(circuit_Totwidth):
    if (len(circuit_Totwidth) % 10 == 0):
        pass
    else:
        circuit_Totwidth = circuit_Totwidth[:len(circuit_Totwidth)-(len(circuit_Totwidth) % 10)]
    
    return circuit_Totwidth
    


class X_Gate(tpm.GenericGate):
    
    def __init__(self, pi_pulse_list):
        # build waveforms
        print(pi_pulse_list)
        #peak_x, sigma, unit = *pi_pulse_list,
        sigmaLen, flat, unit = *pi_pulse_list,
        pi_pulse = ~Wave(
            setFunc(
                'gaussian_square',
                {'sigmaLen': sigmaLen, 'flat':flat},
                 unit))
        null = Waveform(Waveform._nullBlock(pi_pulse.span))
        # build qubitchannels
        gate_seq = ~pi_pulse/~null
        gate_seq.name = 'I'
        # gate_seq2 = ....
        # gate_seq2.name = ''
        # construct gate
        super().__init__(gate_seq)

 
class Y_Gate(tpm.GenericGate):
    
    def __init__(self, pi_pulse_list):
        # build waveforms
        print(pi_pulse_list)
        #peak_x, sigma, unit = *pi_pulse_list,
        sigmaLen, flat, unit = *pi_pulse_list,
        pi_pulse = ~Wave(
            setFunc(
                'gaussian_square',
                {'sigmaLen': sigmaLen, 'flat':flat},
                 unit))
        null = Waveform(Waveform._nullBlock(pi_pulse.span))
        # build qubitchannels
        gate_seq = ~null / ~pi_pulse
        gate_seq.name = 'Q'
        # gate_seq2 = ....
        # gate_seq2.name = ''
        # construct gate
        super().__init__(gate_seq)

class READOUT(tpm.GenericGate):
    
    def __init__(self,readout_pulse_list):
        # build waveforms
        print(readout_pulse_list)
       
        sigmaLen,flat,sigmaLen_m,flat_m, unit= readout_pulse_list
        #sigmaLen,flat, unit= readout_pulse_list
        readout_pulse = ~Wave(
            setFunc(
                'gaussian_square',
                {'sigmaLen': sigmaLen, 'flat':flat},
                 unit))
        null = Waveform(Waveform._nullBlock(readout_pulse.span))
        
        marker_pulse = ~Wave(
            setFunc(
                'gaussian_square',
                {'sigmaLen': sigmaLen_m, 'flat':flat_m},
                 unit))
        
        # build qubitchannels
        #gate_seq = ~null / ~readout_pulse
        gate_seq = ~marker_pulse / ~readout_pulse
        gate_seq.name = 'R'
        # gate_seq2 = ....
        # gate_seq2.name = ''
        # construct gate
        super().__init__(gate_seq)

