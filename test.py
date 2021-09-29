# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 02:34:47 2021

@author: user
"""
# from QuantumCompiler.ShapeModule import setFunc
# from QuantumCompiler.WaveModule import Wave, Waveform
# from QuantumCompiler.TemplateModule import GenericGate
# from QuantumCompiler.QuantumCircuit import QuantumCircuit

from ShapeModule import setFunc
from WaveModule import Wave, Waveform
from TemplateModule import GenericGate
from QuantumCircuit import QuantumCircuit

a = Wave(setFunc('gaussian', {'peak_x': 5e-6, 'sigma': 1e-6}, 10e-6))
b = Waveform(Waveform._nullBlock(a.span*2))
b1 = ~(b+b+~a)
b2 = ~(~a+b+~a+b)
c = ~b / ~a
b1.name = 'b1'
b2.name = 'b2'
c.name = 'c'
# c.plot(allInOne=True)
kk = QuantumCircuit({'a': 0, 'b': 1, 'CC': 2}, 10, ['readout'])
# kk.assign(c, ('a', 0))
# kk.assign(c, ('CC', 0))
# kk.assign(b1, ('b', 8))
# kk.assign(b2, ('b', 5))
kk[('a', 0)] = c
kk[('CC', 0)] = c
kk[('b', 8)] = b1
kk[('b', 5)] = b2

z = GenericGate(c)
z.name = 'some gate'
# kk.assign(z, {'c': ('a', 11)})
# kk.assign(z, {'c': ('readout', 10)})
# kk.assign(z, {'c': ('readout', 9)})
# kk.assign(z, {'c': ('readout', 8)})
kk[('a', 11)] = (z, 'c')
kk[('readout', 10)] = (z, 'c')
kk[('readout', 9)] = (z, 'c')
kk[('readout', 8)] = (z, 'c')



# kk.view()
kk.compile()
# kk.plot()
print(kk@'CC')
