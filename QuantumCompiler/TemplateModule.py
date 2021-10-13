# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 02:01:34 2021

@author: Alaster
"""

from copy import deepcopy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import filedialog, Tk, Frame, Canvas, Scrollbar
from io import BytesIO


# Plotting module
def axis(name='', label='', data=np.array([]), log_bool=False):
    """
    Create an uniform interface to plot data with matplotlib.

    Parameters
    ----------
    name : str, optional
        Name of numpy array. The default is ''.
    label : str, optional
        Name or description of axis. The default is ''.
    data : numpy.array, optional
        Data array. The default is np.array([]).
    log_bool : bool, optional
        Toggle True to enable logarithmic scale. The default is False.

    Returns
    -------
    dict
        A dictionary of settings that to be fed into draw() function.

    """
    return {'name': name, 'label': label, 'data': data, 'log': log_bool}


def draw(
        xdict={}, ydict_list=[],
        figure_name='', titleFontSize=20, size=[6.4, 4.8],
        allInOne=False, toByteStream=False, showSizeInfo=True
        ):
    """
    Formatted plot generation. Dictionary format: dict = {name:str,
    label:str, data:numpy.array, log:bool}.

    Parameters
    ----------
    cls : GenericWave class
        GenericWave class.
    xdict : dict
        Dictionary for x-axis. The default is {}.
    ydict_list : list, optional
        List of y-axis dictionaries with the same format as x-axis
        dictionary. The default is [].
    figure_name : str, optional
        Title of the figure. The default is ''.
    titleFontSize : float, optional
        Font size for the title. The default is 16.
    size : list, optional
        List of subplot sizes in x- and y-axis, respectively. The default
        is [6.4, 4.8].
    allInOne : bool, optional
        Set True to put all traces into the same subplot. The default is False.
    toByteStream : bool, optional
        Set True to convert plot into byte stream without plotting. The default
        is False. Ref: https://www.twblogs.net/a/5eb1097d86ec4d44378845b6
    showSizeInfo : bool, optional
        Set True to show plot size during plot creation. The default is True.

    Returns
    -------
    fig/BytesIO : matplotlib.lines.Line2D or BytesIO object
        Figure of byte stream object.

    """
    if not xdict:
        xdict = axis(data=range(len(ydict_list[0]['data'])))
    num_plot = 1 if allInOne else len(ydict_list)
    fig = plt.figure(figsize=[size[0], size[1] * num_plot])
    fig.suptitle(figure_name, fontsize=titleFontSize, fontweight="bold")
    for i in range(len(ydict_list)):
        if i == 0 or (i > 0 and not allInOne):
            ax = plt.subplot(num_plot, 1, i+1)
        plt.plot(xdict['data'], ydict_list[i]['data'])
        if allInOne and i > 0:  # do not update x,y labels for allInOne mode
            continue
        plt.xlabel(xdict['label'])
        plt.ylabel(ydict_list[i]['label'])
        if xdict['log']:
            ax.set_xscale('log')
        if ydict_list[i]['log']:
            ax.set_yscale('log')
        if allInOne:    # do not update legend for allInOne mode
            continue
        ax.legend([ydict_list[i]['name']], loc="best")
    if allInOne:
        ax.legend([ydict['name'] for ydict in ydict_list], loc="best")
    if showSizeInfo:
        print("plot size=[" + str(size[0]) + "," + str(size[1]) + "]")
    fig.tight_layout()
    if toByteStream:
        byte_data = BytesIO()
        plt.savefig(byte_data)  # save plot to byte stream
        plt.close()  # close plot to disable showing plot
        return byte_data
    plt.show()
    return fig


# Window module
def simple_scrollable_window(windowSize='800x600'):
    w = Tk()
    w.attributes("-topmost", True)
    w.geometry(windowSize)
    cvs = Canvas(w)
    # set up scroall bar
    scrol_y = Scrollbar(w, orient='vertical', command=cvs.yview)
    scrol_y.pack(fill='y', side='right')
    scrol_x = Scrollbar(w, orient='horizontal', command=cvs.xview)
    scrol_x.pack(fill='x', side='bottom')
    # create image grid
    fm = Frame(cvs)

    def run():
        cvs.create_window(0, 0, anchor='nw', window=fm)
        cvs.update_idletasks()
        cvs.configure(
            scrollregion=cvs.bbox('all'),
            xscrollcommand=scrol_x.set, yscrollcommand=scrol_y.set
            )
        cvs.pack(fill='both', expand=True, side='left')
        w.mainloop()
    return w, fm, run


# Storage module
def save(ext, *args):
    """
    Save objects to files with specified extension.

    Parameters
    ----------
    ext : str
        File extension.
    *args : Storable class
        Instances belong to any children class of Storable.

    """
    for i, obj in enumerate(args):
        if obj.name == '':
            obj.name = input(
                f'Empty name string for {i}th item, set object name:'
                )
        with open(f'{obj.name}' + ext, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load(ext, *args):
    """
    Load object from files with 2 methods: The 'dialog' mode using a dialog
    box to import filenames while 'arg' mode using file names given in
    *args. The mode is specified by ext variable.

    Parameters
    ----------
    cls : Storable class
        Class of instances with 'name' attribute.
    ext : str
        Determine whether loading file using dialog box or straightforward
        filenames in *args:
            'dialog' mode   : ext=''
            'arg' mode      : ext='.[extension]'
    *args : String
        Filenames to be imported when using 'arg' mode.

    Returns
    -------
    object
        Loaded objects in tuple.

    """
    if ext:
        args = get_path(ext, 'Select object files')

    objList = []
    for filename in args:
        with open(filename, 'rb') as f:
            objList += [pickle.load(f)]
    return *objList,


def get_path(ext='', title='Select item'):
    """
    Get absolute path using dialogue box.

    Parameters
    ----------
    cls : Storable
        Storable class.
    ext : str, optional
        File extension. The default is ''.
    title : Box title, optional
        Box title. The default is 'Select item'.

    Returns
    -------
    tuple
        tuple of absolute paths.

    """
    w = tkinter.Tk()
    w.withdraw()
    w.attributes("-topmost", True)
    return filedialog.askopenfilenames(
        parent=w,
        filetypes=[(ext.upper() + ' Files', ext)],
        title=title)


class Comparables():
    EFF_FREQ_DIGIT = 5
    EFF_TIME_DIGIT = 0 + 9

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name=''):
        self._name = name

    @property
    def span(self):
        return round(
            self.x[-1] - self.x[0], self.__class__.EFF_TIME_DIGIT
            )

    @property
    def x(self):
        return self._x

    @property
    def dx(self):
        try:
            return round(self.x[1] - self.x[0], self.__class__.EFF_TIME_DIGIT)
        except(IndexError):
            return 0

    @property
    def df(self):
        try:
            return round(1 / self.dx, self.__class__.EFF_FREQ_DIGIT)
        except(ZeroDivisionError):
            print('unable to define frequency axis wirh 1 point')
            return 0

    @property
    def xaxis(self):
        return axis('t', 'time (s)', self.x, False)

    def __len__(self):
        """
        Get number of points in the object. Denoted as len(self).

        Returns
        -------
        int
            Number of points in the object.

        """
        return len(self.x)

    def __max__(self, *gwObj):
        """
        Return object contains the largest number of points. Denoted as
        max(self, *gwObj).

        Parameters
        ----------
        *gwObj : GenericWave
            Multiple objects to be compared.

        Returns
        -------
        GenericWave
            Largest object.

        """
        return max(self, *gwObj, key=len)

    def __min__(self, *gwObj):
        """
        Return object contains the smallest number of points. Denoted as
        min(self, *gwObj).

        Parameters
        ----------
        *gwObj : GenericWave
            Multiple objects to be compared.

        Returns
        -------
        GenericWave
            Smallest object.

        """
        return min(self, *gwObj, key=len)

    def __lt__(self, gwObj):
        """
        Operator method for self < gwObj.

        Parameters
        ----------
        gwObj : GenericWave
            Operand.

        Returns
        -------
        Boolean
            True if len(self) < len(gwObj).

        """
        return len(self) < len(gwObj)

    def __le__(self, gwObj):
        """
        Operator method for self <= gwObj.

        Parameters
        ----------
        gwObj : GenericWave
            Operand.

        Returns
        -------
        Boolean
            True if len(self) <= len(gwObj).

        """
        return len(self) <= len(gwObj)

    def __eq__(self, gwObj):
        """
        Operator method for self == gwObj.

        Parameters
        ----------
        gwObj : GenericWave
            Operand.

        Returns
        -------
        Boolean
            True if len(self) == len(gwObj).

        """
        return len(self) == len(gwObj)

    def __ne__(self, gwObj):
        """
        Operator method for self != gwObj.

        Parameters
        ----------
        gwObj : GenericWave
            Operand.

        Returns
        -------
        Boolean
            True if len(self) != len(gwObj).

        """
        return len(self) != len(gwObj)

    def __ge__(self, gwObj):
        """
        Operator method for self >= gwObj.

        Parameters
        ----------
        gwObj : GenericWave
            Operand.

        Returns
        -------
        Boolean
            True if len(self) >= len(gwObj).

        """
        return len(self) >= len(gwObj)

    def __gt__(self, gwObj):
        """
        Operator method for self > gwObj.

        Parameters
        ----------
        gwObj : GenericWave
            Operand.

        Returns
        -------
        Boolean
            True if len(self) > len(gwObj).

        """
        return len(self) > len(gwObj)


class GenericWave(Comparables):

    @property
    def y(self):
        return self._y

    def __matmul__(self, xList=[]):
        """
        Return the corresponding y at the indicated x by interpolation. Denoted
        as self @ x.

        Parameters
        ----------
        xList : list
            List of position x.

        Returns
        -------
        list
            List of amplitude y.

        """
        return np.array([np.interp(x, self.x, self.y) for x in xList])

    def plot(self, figure_name='', toByteStream=False):
        """
        Plot y v.s. x.

        Parameters
        ----------
        figure_name : TYPE, optional
            DESCRIPTION. The default is ''.
        toByteStream : bool, optional
            Set True to convert plot into byte stream without plotting. The
            default is False.

        Returns
        -------
        matplotlib.lines.Line2D
            Plot handler object.

        """
        xdict = self.xaxis
        ydict_list = [{
            'name': self.name, 'label': 'amplitude',
            'data': self.y, 'log': False
            }]
        return draw(xdict, ydict_list, figure_name, toByteStream=toByteStream)


class GenericGate():

    def __init__(self, *qcObj):
        temp = deepcopy(qcObj)
        temp = temp[0].__class__.alignQubitChannels(*temp)
        self._qubitDict = {qcObj0.name: qcObj0 for qcObj0 in temp}
        self._name = ''

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name=''):
        self._name = name

    @property
    def numOfQubits(self):
        return len(self._qubitDict)

    @property
    def qubitNames(self):
        return *self._qubitDict.keys(),

    @property
    def qubitDict(self):
        return self._qubitDict

    def __str__(self):
        """
        Print the status of a Gate object.

        Returns
        -------
        string
            The status of a Gate object.

        """
        return f"name: {self.name}\n" + \
            f"gate type: {self.numOfQubits}-qubit gate\n" + \
            f"qubit names: {list(self._qubitDict.keys())}\n" + \
            f"ID: {id(self)}"

    def __setitem__(self, qbname, qbcObj):
        """
        Assign a QubitChannel object to a qubit according to a specified name.

        Parameters
        ----------
        qbname : str
            Name of qubit.
        qbcObj : QubitChannel
            QubitChannel object.

        """

        self._qubitDict[qbname] = qbcObj

    def __getitem__(self, qbname):
        """
        Return a QubitChannel object according to a specified name.

        Parameters
        ----------
        qbname : str
            Name of qubit.

        Returns
        -------
        QubitChannel
            QubitChannel object.

        """
        return self._qubitDict[qbname]
