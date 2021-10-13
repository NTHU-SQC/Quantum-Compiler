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

from .WaveModule import QubitChannel
from .TemplateModule import simple_scrollable_window, Namables


class QuantumCircuit(Namables):

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

    def view(self, compiled=False, windowSize='800x600'):
        """
        Show the current diagram in grid plot.

        Parameters
        ----------
        compiled : boolean, optional
            Set True to view the compiled diagram, uncompiled one otherwise.
            The default is False.
        windowSize : str, optional
            Window size that is specified in string. The default is '800x600'.

        """
        # check if compiled and select diagram
        if not hasattr(self, 'compiled'):
            print('Warning: The object has not compiled yet')
            print('Show the uncompiled diagram')
            compiled = False
        diagram = self.diagram
        if compiled:
            diagram = self.compildDiagram
        # create vectorized functions
        f = np.vectorize(
            lambda x: x.__str__() if isinstance(x, QubitChannel) else str(
                np.nan), otypes=[object]
            )
        g = np.vectorize(
            lambda x: 'Gate: ' + x + '\n' if x else '', otypes=[object]
            )
        # convert the diagram into info strings
        str1 = g(self.gateName)
        str2 = f(diagram)
        data = str1 + str2
        namefield = np.array([['']] * len(diagram[:, 0]), dtype=object)
        for key, val in {**self.qubitDict, **self.auxiliaryDict}.items():
            namefield[val] = key + f':{val}'
        data = np.hstack([namefield, data])
        timeindex = np.array(
            [''] + list(range(len(diagram[0, :]))), dtype=object
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
            fm, text='View assignment', command=lambda: self.view(
                compiled=True)
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

    def compile(self):
        """
        Compile the quantum circuit.

        """
        self.compildDiagram = deepcopy(self.diagram)
        f = np.vectorize(lambda x: isinstance(x, QubitChannel))
        table = f(self.compildDiagram)
        col_bool = np.bitwise_or.reduce(table, axis=1)
        # filter nan in 'qubit' direction
        if not np.bitwise_and.reduce(col_bool):
            raise ValueError('Found unassigned qubit')
        # filter nan in 'time' direction
        row_bool = np.bitwise_or.reduce(table, axis=0)
        diagram = self.compildDiagram[:, row_bool]
        table = table[:, row_bool]
        # align QubitChannel objects in the table column by column
        for time_idx in range(len(table[0, :])):
            QubitChannel.alignQubitChannels(*diagram[
                table[:, time_idx], time_idx
                ])
            # diagram[table[:, time_idx], time_idx
            #         ] = QubitChannel.alignQubitChannels(
            #             *diagram[table[:, time_idx], time_idx]
            #             )
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
        Get the compiled QubitChannel object y arrays with the specified name.

        Parameters
        ----------
        qubit : str, int
            Index or the name of qubit.

        Returns
        -------
        list
            Return a list of y arrays of the compiled QubitChannel object.

        """
        return self.compiled[self.get_index(qubit)].y

    def __setitem__(self, idxTuple, item):
        """
        Assign the Gate/QubitChannel object to the specified index tuple. This
        is an alternative form for assign() method. Note that this method only
        can assign 1 QubitChannel object at a time.

        Parameters
        ----------
        idxTuple : tuple/list
            Index tuple for the assignment on the diagram.
        item : tuple/QubitChannel
            Item to be assigned. For tuple (Gate) mode takes the form:
                (Gate, qubit name)

        """
        qubit = self.get_index(idxTuple[0])
        time = idxTuple[1]
        if isinstance(item, tuple) or isinstance(item, list):
            self.assign(item[0], {item[1]: (qubit, time)})
        else:
            self.assign(item, (qubit, time))

    def __getitem__(self, idxTuple):
        """
        Get QubitChannel object from diagram with specified index tuple.

        Parameters
        ----------
        idxTuple : tuple/list
            Index tuple for the diagram element accessing.

        Returns
        -------
        QubitChannel
            Corresponding QubitChannel object.

        """
        qubit = self.get_index(idxTuple[0])
        time = idxTuple[1]
        return self.diagram[qubit, time]
