# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 19:34:12 2021

Ref:
set window size:
    https://pythonexamples.org/python-tkinter-set-window-size/
embedding matplotlib plot inside tkinter label:
    https://stackoverflow.com/questions/67648380
scrolling a group of widgets:
    https://sodocumentation.net/tkinter/topic/8931/scrolling-widgets

Bug:
1.tkinter "_tkinter.TclError: image "pyimage9" doesn't exist":
    https://stackoverflow.com/questions/25460418

@author: Alaster
"""

import numpy as np
from copy import deepcopy
from tkinter import Label, Button
from PIL import ImageTk, Image

from WaveModule import QubitChannel
from TemplateModule import save, load, simple_scrollable_window


class QuantumCircuit(object):

    def __init__(self, qubit={}, blockNum=1, auxiliary={}):
        """
        Create a timeline of quantum circuit diagram.

        Parameters
        ----------
        qubit : dict/list, optional
            Qubit control channel assignment. For dict type the argument uses
            {qubit name : qubit index, ...} format, while for list type is
            [qubit name, ...] as the indices have the same order as the list.
            The default is {}.
        blockNum : int, optional
            Number of time indices. The default is 1.
        auxiliary : dict/list, optional
            Similar to qubit for special purposes. The index order can be mixed
            with qubit in dict mode while the in list mode the index order is
            always later than qubit ones. The default is {}.

        """
        self.diagram = np.asarray(
            [[np.nan] * blockNum] * (len(qubit) + len(auxiliary)),
            dtype=object
            )
        # check datatype and assign
        if isinstance(qubit, list):
            self._qubitDict = dict(zip(qubit, range(len(qubit))))
        elif isinstance(qubit, dict):
            self._qubitDict = qubit
        else:
            raise TypeError('qubitDict: Unsupported format')
        if isinstance(auxiliary, list):
            used_idx = [val for _, val in self._qubitDict.items()]
            unused_idx = sorted([
                val for val in range(len(auxiliary) + len(qubit))
                if val not in used_idx
                ])
            self._auxiliaryDict = dict(zip(auxiliary, unused_idx))
        elif isinstance(auxiliary, dict):
            self._auxiliaryDict = auxiliary
        else:
            raise TypeError('auxiliaryDict: Unsupported format')
        # index check
        for key, val in {**self._qubitDict, **self._auxiliaryDict}.items():
            if val >= len(self.diagram[:, 0]):
                raise ValueError(
                    f'QubitChannel \'{key}\' assignment out of bound with ' +
                    f'index: {val}'
                        )
        self._name = ''
        self.gateName = np.asarray(
            [[''] * blockNum] * (len(qubit) + len(auxiliary)),
            dtype=object
            )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name=''):
        self._name = name

    @property
    def qubitDict(self):
        return self._qubitDict

    @property
    def auxiliaryDict(self):
        return self._auxiliaryDict

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
        copied = deepcopy(gateObj)
        width = len(self.diagram[:, 0])
        if isinstance(mapping, dict):
            # Gate assign mode
            for key, idx_tag in mapping.items():
                qubitIdx = self.get_index(idx_tag[0])
                blockNum = idx_tag[1] - len(self.diagram[0, :]) + 1
                if blockNum > 0:
                    self.diagram = np.concatenate((
                        self.diagram, np.asarray(
                            [[np.nan] * blockNum] * width, dtype=object
                            )
                        ), axis=1)
                    self.gateName = np.concatenate((
                        self.gateName, np.asarray(
                            [[''] * blockNum] * width, dtype=object
                            )
                        ), axis=1)
                self.diagram[qubitIdx, idx_tag[1]] = copied.qubitDict[key]
                self.gateName[qubitIdx, idx_tag[1]] = copied.name
        else:
            # QubitChannel assign mode
            qubitIdx = self.get_index(mapping[0])
            blockNum = mapping[1] - len(self.diagram[0, :]) + 1
            if blockNum > 0:
                self.diagram = np.concatenate((
                    self.diagram, np.asarray(
                        [[np.nan] * blockNum] * width, dtype=object
                        )
                    ), axis=1)
            self.diagram[qubitIdx, mapping[1]] = copied

    def get_index(self, qubit_name):
        """
        Get the index of the qubit/auxiliary according to its name.

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
            return self.auxiliaryDict[qubit_name]

    def view(self, windowSize='800x600'):
        """
        Show the current diagram in grid plot.

        Parameters
        ----------
        windowSize : str, optional
            Window size that is specified in string. The default is '800x600'.

        """
        f = np.vectorize(
            lambda x: x.__str__() if isinstance(x, QubitChannel) else str(
                np.nan), otypes=[object]
            )
        g = np.vectorize(
            lambda x: 'Gate: ' + x + '\n' if x else '', otypes=[object]
            )
        str1 = g(self.gateName)
        str2 = f(self.diagram)
        data = str1 + str2
        namefield = np.array([['']] * len(self.diagram[:, 0]), dtype=object)
        for key, val in {**self.qubitDict, **self.auxiliaryDict}.items():
            namefield[val] = key + f':{val}'
        data = np.hstack([namefield, data])
        timeindex = np.array(
            [''] + list(range(len(self.diagram[0, :]))), dtype=object
            )
        data = np.array(np.vstack([timeindex, data]), dtype=object)
        # create a scrollable window
        _, fm, run = simple_scrollable_window(windowSize)
        for i, row in enumerate(data):
            for j, item in enumerate(row):
                Label(
                    fm, text=item, font='Consolas',
                    relief='solid', borderwidth=1
                    ).grid(row=i, column=j, ipadx=5, ipady=5, sticky='news')
        run()

    def plot(self, windowSize='800x600'):
        """
        Show the compiled waveform plots.

        Parameters
        ----------
        windowSize : str, optional
            Window size that is specified in string. The default is '800x600'.

        """
        if not hasattr(self, 'compiled'):
            raise RuntimeError('The object has not compiled yet')
        # create a scrollable window
        _, fm, run = simple_scrollable_window(windowSize)
        Button(
            fm, text='View assignment', command=self.view
            ).grid(row=0, column=0, columnspan=2)
        count = 1
        img_ref = []
        for key, val in {**self.qubitDict, **self.auxiliaryDict}.items():
            Label(
                fm, text=key + f':{val}', font='Consolas',
                relief='solid', borderwidth=1
                ).grid(row=count, column=0, ipadx=5, ipady=5, sticky='news')
            img_data = self.compiled[val].plot(
                allInOne=False, toByteStream=True, showSizeInfo=False,
                size=[20, 4]
                )
            render = ImageTk.PhotoImage(Image.open(img_data))
            img_ref += [render]
            img = Label(fm, image=render, borderwidth=1, relief='solid')
            img.grid(row=count, column=1, ipadx=5, ipady=5, sticky='news')
            img.image = render
            count += 1
        run()

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
        for qubit_idx, row in enumerate(table):
            for time_idx, flag in enumerate(row):
                if flag:
                    continue
                span_idx = np.where(f(diagram[:, time_idx]))[0][0]
                wire_idx = np.where(f(diagram[qubit_idx, :]))[0][0]
                diagram[qubit_idx, time_idx] = QubitChannel.null(
                    diagram[span_idx, time_idx], diagram[qubit_idx, wire_idx]
                    )
        try:
            self.compiled = np.sum(diagram, axis=1)
        except SystemError:
            raise ValueError('Error during wire concatenation')

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
        save('.qckt', *args)

    @classmethod
    def load(cls, *args):
        return load('.qckt', *args)


if __name__ == '__main__':
    # from shape_functionV5 import gaussian, get_x
    from ShapeModule import setFunc
    from WaveModule import Wave, Waveform
    from TemplateModule import GenericGate
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
    kk.assign(c, ('a', 0))
    kk.assign(c, ('CC', 0))
    kk.assign(b1, ('b', 8))
    kk.assign(b2, ('b', 5))
    z = GenericGate(c)
    z.name = 'some gate'
    kk.assign(z, {'c': ('a', 11)})
    kk.assign(z, {'c': ('readout', 10)})
    kk.assign(z, {'c': ('readout', 9)})
    kk.assign(z, {'c': ('readout', 8)})
    # kk.view()
    kk.compileCkt()
    kk.plot()
    # print(kk@'c')
    pass
