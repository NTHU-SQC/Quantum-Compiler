# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 00:59:00 2021

@author: cluster
"""

import sys
import numpy as np
sys.path.append('../')
from time import sleep
from Quantum_Circuit_Editor.Gate_Design import *
from instruments.core.jsonIO import jsonFormat
from instruments.drivers.Tektronix.Tektronix_AWG import AWG5208

# X Gate
gate_paraX = [5e-9,6e-8,10e-8]
X =  X_Gate(gate_paraX)
X.name='X_gate'
print(X)


# Y Gate
gate_paraY = [5e-9,6e-8,10e-8]
Y =  Y_Gate(gate_paraY)
Y.name='Y_gate'
print(Y)


#virtual Z gate Swap X and Y
gate_swapX = [5e-9,6e-8,10e-8]
swapX = Y_Gate(gate_swapX)
swapX.name='Virtual_Z X_gate'
print(swapX)


gate_swapY = [5e-9,6e-8,10e-8]
swapY = X_Gate(gate_swapY)
swapY.name='Virtual_Z Y_gate'
print(swapY)


# Readout/Measurement

gate_paraREADOUT = [3e-8,5e-6,6e-9,1e-7,10e-6]
readout_ = READOUT(gate_paraREADOUT)
readout_.name='READOUT'
print(readout_)

gate_paraMarker = []

Clifford_Length = 20
#Clifford_gate = {"1":"X","2":"Y"}
Clifford_gate = {"1":"X","2":"Y","3":"Z"}
#Clifford_gate_Swap = {""}


#kk = QuantumCircuit({'qbit1': 0}, 2, ['readout'])
kk = QuantumCircuit({'qbit1': 0}, Clifford_Length,['readout'])


gate_list = []

for index in range(Clifford_Length):
    #rand = random.randint(0, 10)
    rand = random.randint(1,3)
    result = Clifford_gate[str(rand)]
    gate_list.append(result)

print(gate_list)


# Assign Waveform
ngate = 0 
while (ngate < len(gate_list)):
    if (ngate == (len(gate_list)-1)):
        if (gate_list[ngate] == "X"):
            kk.assign(X, {'I' : ('qbit1', ngate)})
            print("X_gate")
        elif (gate_list[ngate] == "Y"):
            kk.assign(Y, {'Q' : ('qbit1', ngate)})
            print("Y_gate")

    elif(ngate < (len(gate_list) - 1)):
        if (gate_list[ngate+1] == "Z"):
            if (gate_list[ngate] == "X"):
                kk.assign(swapX, {'Q' : ('qbit1', ngate)})
                print("swapX")
            elif (gate_list[ngate] == "Y"):
                kk.assign(swapY, {'I' : ('qbit1', ngate)})
                print("swapY")
        else:
            if (gate_list[ngate] == "X"):
                kk.assign(X, {'I' : ('qbit1', ngate)})
                print("X_gate")
            elif (gate_list[ngate] == "Y"):
                kk.assign(Y, {'Q' : ('qbit1', ngate)})
                print("Y_gate")
    ngate += 1


    #print("last_result:" + str(last_result))
kk.assign(readout_, {'R' : ('readout', Clifford_Length)})

print(X.qubitNames)
print(Y.qubitNames)
#kk.view()

#Compiling the Gate Sequence
kk.compileCkt()


print("total waveform length:" + str(len(kk.compiled[0].x)))
kk.plot()

wfm = dispose_point(kk.compiled[0].x)

# connect to AWG
awg = AWG5208(
    inst_name='AWG5208',
    inst_address='TCPIP0::192.168.20.43::inst0::INSTR')
# set sampling rate and reference clock
awg.set_sample_rate(sample_rate=1.0E9)
awg.set_extref_source(ref_freq=10E6)
# clear waveforms and sequences
awg.clr_wfm()
awg.clr_seq()
print(len(wfm))
kk.view()


# get waveform from Randomized Benchmarking Simulator 
# Removing the additional point in unit digit

awg.set_wfm(
    wfm_name='qbit1_I',
    wfm= dispose_point(kk.compiled[0].y[0])
   )
awg.set_wfm(
    wfm_name='qbit1_Q',
    wfm= dispose_point(kk.compiled[0].y[1])
   )
awg.set_wfm(
    wfm_name='Readout',
    wfm= dispose_point(kk.compiled[1].y[1]),
    mkr1= dispose_point(kk.compiled[1].y[0])
   )
# upload to awg
awg.upload_wfm()

# assign waveforms/sequences to channels
awg.assign_ch(1, 'qbit1_I')
awg.assign_ch(2, 'Readout', auto_output=False)
awg.assign_ch(4, 'qbit1_Q')

# set channel amplitudes
awg.set_ch_amp(
    ch=1,
    wfm_Vpp=1.0, wfm_offset=0.,
    mkr1=1.0, mkr2=1.0, mkr3=1.0, mkr4=1.0)
awg.set_ch_amp(
    ch=2,
    wfm_Vpp=1.0, wfm_offset=0.,
    mkr1=1.0, mkr2=1.0, mkr3=1.0, mkr4=1.0)
