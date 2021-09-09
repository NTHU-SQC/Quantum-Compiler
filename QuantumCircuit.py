# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 19:34:12 2021

@author: Alaster
"""

import numpy as np
import pickle
from WaveModule import QubitChannel


class QuantumCircuit(object):

    def __init__(self, qubitNames=[], blockNum=1, readoutNames=[]):
        self.diagram = np.asarray(
            [[np.nan] * blockNum] * (len(qubitNames) + len(readoutNames)),
            dtype=object
            )
        self._qubitNames = qubitNames
        self._readoutNames = readoutNames
        self._name = ''

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name=''):
        self._name = name

    @property
    def qubitNames(self):
        return self._qubitNames

    @qubitNames.setter
    def qubitNames(self, qubitNames=[]):
        if len(self._qubitNames) != len(qubitNames):
            raise ValueError('Name field size mismatched')
        self._qubitNames = qubitNames

    @property
    def readoutNames(self):
        return self._readoutNames

    @readoutNames.setter
    def readoutNames(self, readoutNames=[]):
        if len(self._readoutNames) != len(readoutNames):
            raise ValueError('Name field size mismatched')
        self._readoutNames = readoutNames

    def assign(self, gateObj, mapping):
        """
        Assign the Gate object to the specified index.

        Parameters
        ----------
        gateObj : Gate or QubitChannel
            Gate or QubitChannel object.
        mapping : dict or tuple
            The datatype determines how the gateObj is assigned:
                dict -> name-index pair for Gate object assignment. The 'name'
                    key is the name of the QubitChannel object and the 'index'
                    value is the corresponding indices in the circuit diagram.
                tuple -> indices for single QubitChannel object assignment.

        Returns
        -------
        None.

        """
        if isinstance(mapping, dict):
            for key, idx in mapping.items():
                blockNum = idx - len(self.diagram[0, :]) + 1
                if blockNum > 0:
                    self.diagram = np.concatenate((
                        self.diagram, np.asarray(
                            [[np.nan] * blockNum] * (
                                len(self._qubitNames) + len(self._readoutNames)
                                )
                            )
                        ), axis=1)
                self.diagram[idx] = gateObj._qubitDict[key]
        else:
            blockNum = mapping[1] - len(self.diagram[0, :]) + 1
            if blockNum > 0:
                self.diagram = np.concatenate((
                    self.diagram, np.asarray(
                        [[np.nan] * blockNum] * (
                            len(self._qubitNames) + len(self._readoutNames)
                            )
                        )
                    ), axis=1)
            self.diagram[mapping] = gateObj

    def get_qubit_index(self, qubit_name):
        """
        Get the index of the qubit according to its name.

        Parameters
        ----------
        qubit_name : str
            Name of the qubit.

        Returns
        -------
        int
            Qubit index.

        """
        try:
            return self._qubitNames.index(qubit_name)
        except ValueError:
            return self._readoutNames.index(qubit_name)

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
        # replace nans with null QubitChannel objects
        for qubit_idx in range(len(diagram[:, 0])):
            for time_idx in range(len(diagram[0, :])):
                if table[qubit_idx, time_idx]:
                    continue
                span_idx = np.where(f(diagram[qubit_idx, :]))[0][0]
                wire_idx = np.where(f(diagram[:, time_idx]))[0][0]
                diagram[qubit_idx, time_idx] = QubitChannel.null(
                    diagram[qubit_idx, span_idx], diagram[wire_idx, time_idx]
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
            qubit = self.get_qubit_index(qubit)
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
    a = Wave(gaussian, [get_x(10e-6), 5e-6, 1e-6])
    b = Waveform(Waveform._nullBlock(a.span*2))
    c = ~b / ~a
    kk = QuantumCircuit(['a', 'b', 'c'], 10)
    kk.assign(c, (0, 0))
    kk.assign(c, (1, 0))
    # kk.assign(c, (2, 0))
    kk.assign(c, (0, 5))
    kk.assign(c, (2, 5))

    kk.compileCkt()
    pass
