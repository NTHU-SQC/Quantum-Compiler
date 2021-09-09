# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 19:34:12 2021

@author: Alaster
"""

import numpy as np
import pickle
from WaveModule import QubitChannel


class QuantumCircuit(object):

    def __init__(self, qubit={}, blockNum=1, readout={}):
        self.diagram = np.asarray(
            [[np.nan] * blockNum] * (len(qubit) + len(readout)),
            dtype=object
            )
        # check datatype and assign
        if isinstance(qubit, list):
            self._qubitDict = dict(zip(qubit, range(len(qubit))))
        elif isinstance(qubit, dict):
            self._qubitDict = qubit
        else:
            raise TypeError('qubitDict: Unsupported format')
        if isinstance(readout, list):
            self._readoutDict = dict(
                zip(readout, np.arange(len(readout)) + len(qubit))
                )
        elif isinstance(readout, dict):
            self._readoutDict = readout
        else:
            raise TypeError('readoutDict: Unsupported format')
        # index check
        for key, val in {**self._qubitDict, **self._readoutDict}.items():
            if val > len(self.diagram[:, 0]):
                raise ValueError(
                    'Found ' + str(key) + ' out of bound: ' + str(val)
                    )
        self._name = ''

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name=''):
        self._name = name

    @property
    def qubitDict(self):
        return self._qubitDict

    @qubitDict.setter
    def qubitDict(self, qubit={}):
        if len(self._qubitDict) != len(qubit):
            raise ValueError('Name field size mismatched')
        self._qubitDict = qubit

    @property
    def readoutDict(self):
        return self._readoutDict

    @readoutDict.setter
    def readoutDict(self, readout={}):
        if len(self._readoutDict) != len(readout):
            raise ValueError('Name field size mismatched')
        self._readoutDict = readout

    def assign(self, gateObj, mapping):
        """
        Assign the Gate object to the specified index.

        Parameters
        ----------
        gateObj : Gate or QubitChannel
            Gate or QubitChannel object.
        mapping : dict or list
            The datatype determines how the gateObj is assigned:
                dict -> name-index pair for Gate object assignment. The 'name'
                    key is the name of the QubitChannel object and the 'index'
                    value is the corresponding indices in the circuit diagram.
                list -> indices for single QubitChannel object assignment.

        Returns
        -------
        None.

        """
        if isinstance(mapping, dict):
            for key, idx_tag in mapping.items():
                qubitIdx = self.get_index(idx_tag[0])
                blockNum = idx_tag[1] - len(self.diagram[0, :]) + 1
                if blockNum > 0:
                    self.diagram = np.concatenate((
                        self.diagram, np.asarray(
                            [[np.nan] * blockNum] * len(self.diagram[:, 0])
                            )
                        ), axis=1)
                self.diagram[qubitIdx, idx_tag[1]] = gateObj._qubitDict[key]
        else:
            qubitIdx = self.get_index(mapping[0])
            blockNum = mapping[1] - len(self.diagram[0, :]) + 1
            if blockNum > 0:
                self.diagram = np.concatenate((
                    self.diagram, np.asarray(
                        [[np.nan] * blockNum] * len(self.diagram[:, 0])
                        )
                    ), axis=1)
            self.diagram[qubitIdx, mapping[1]] = gateObj

    def get_index(self, qubit_name):
        """
        Get the index of the qubit/readout according to its name.

        Parameters
        ----------
        qubit_name : str, int
            Name of the qubit. Return the input if the input is an int.

        Returns
        -------
        int
            Qubit index.

        """
        if isinstance(qubit_name, int):
            return qubit_name
        try:
            return self.qubitDict[qubit_name]
        except KeyError:
            return self.readoutDict[qubit_name]

    def compileCkt(self):
        """
        Compile the quantum circuit.

        """
        f = np.vectorize(lambda x: isinstance(x, QubitChannel))
        table = f(self.diagram)
        col_bool = np.bitwise_or.reduce(table, axis=1)
        # filter nan in 'qubit' direction
        if not np.bitwise_and.reduce(col_bool):
            raise ValueError('Found unassigned qubit')
        # filter nan in 'time' direction
        row_bool = np.bitwise_or.reduce(table, axis=0)
        diagram = self.diagram[:, row_bool]
        table = table[:, row_bool]
        # align QubitChannel objects in the table column by column
        for time_idx in range(len(table[0, :])):
            diagram[table[:, time_idx], time_idx
                    ] = QubitChannel.alignQubitChannels(
                        *diagram[table[:, time_idx], time_idx]
                        )
        # replace nans with null QubitChannel objects
        for qubit_idx in range(len(diagram[:, 0])):
            for time_idx in range(len(diagram[0, :])):
                if table[qubit_idx, time_idx]:
                    continue
                span_idx = np.where(f(diagram[:, time_idx]))[0][0]
                wire_idx = np.where(f(diagram[qubit_idx, :]))[0][0]
                diagram[qubit_idx, time_idx] = QubitChannel.null(
                    diagram[span_idx, time_idx], diagram[qubit_idx, wire_idx]
                    )
        self.compiled = np.sum(diagram, axis=1)

    def __matmul__(self, qubit):
        """
        Return the compiled QubitChannel amplitude data y.

        Parameters
        ----------
        qubit : str, int
            Index or the name of qubit.

        Returns
        -------
        list
            List of y from each wire in the compiled QubitChannel object.

        """
        if isinstance(qubit, str):
            qubit = self.get_index(qubit)
        return self.compiled[qubit].y

    @classmethod
    def save(cls, *args):
        """
        Save QuantumCircuit object to .qckt files.

        Parameters
        ----------
        cls : QuantumCircuit class
            QuantumCircuit class.
        *args : QuantumCircuit
            Object to be saved.

        """
        for i, gwObj in enumerate(args):
            if gwObj.name == '':
                gwObj.name = input(
                    'Empty name string for {i}th item, set object name:'
                    )
            with open(f'{gwObj.name}.qckt', 'wb') as f:
                pickle.dump(gwObj, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, *args):
        """
        Load QuantumCircuit object from .qckt files.

        Parameters
        ----------
        cls : QuantumCircuit class
            QuantumCircuit class.
        *args : String
            Filename.

        Returns
        -------
        QuantumCircuit
            Loaded objects.

        """
        objList = []
        for filename in args:
            with open(filename, 'rb') as f:
                objList += [pickle.load(f)]
        return *objList,


if __name__ == '__main__':
    from shape_functionV4 import gaussian, get_x
    from WaveModule import Wave, Waveform
    from TemplateModule import GenericGate
    a = Wave(gaussian, [get_x(10e-6), 5e-6, 1e-6])
    b = Waveform(Waveform._nullBlock(a.span*2))
    b1 = ~(b+b+~a)
    b2 = ~(~a+b+~a+b)
    c = ~b / ~a
    c.name = 'c'
    # kk = QuantumCircuit({'a': 0, 'b': 1, 'c': 2}, 10, ['readout'])
    kk = QuantumCircuit(['a', 'b', 'c'], 10, ['readout'])
    kk.assign(c, (0, 0))
    kk.assign(c, (1, 0))
    z = GenericGate(c)
    kk.assign(z, {'c': (0, 11)})
    kk.assign(b1, (0, 5))
    kk.assign(b2, (2, 5))
    kk.assign(z, {'c': ('readout', 10)})
    kk.assign(z, {'c': ('c', 9)})
    kk.assign(z, {'c': ('c', 8)})

    kk.compileCkt()
    print(kk@'c')
    pass
