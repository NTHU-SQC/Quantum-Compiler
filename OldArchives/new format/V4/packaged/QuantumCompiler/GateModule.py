# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 20:18:43 2021

for set / get vs IDE:
    https://stackoverflow.com/questions/52312897

@author: Alaster
"""

from numpy import *

from .ShapeModule import *
from .TemplateModule import GenericGate
from .WaveModule import *


class Gate(GenericGate):

    def __init__(self, parameters):
        # build waveforms
        
        # build qubitchannels
        
        # construct gate
        super().__init__()
