# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 02:01:34 2021

@author: Alaster
"""

from scipy.signal import welch
from scipy.fft import fft, ifft, fftfreq, fftshift
from copy import deepcopy
import pickle
import numpy as np
import matplotlib.pyplot as plt


def axis(name='', label='', data=np.array([]), log_bool=False):
    return {'name': name, 'label': label, 'data': data, 'log': log_bool}


def draw(xdict={}, ydict_list=[], figure_name='', titleFontSize=20,
         size=[6.4, 4.8]):
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

    Returns
    -------
    fig : matplotlib.lines.Line2D
        Figure object.

    """
    if not xdict:
        xdict = axis(data=range(len(ydict_list[0]['data'])))
    num_plot = len(ydict_list)
    fig = plt.figure(figsize=[size[0], size[1] * num_plot])
    fig.suptitle(figure_name, fontsize=titleFontSize, fontweight="bold")
    for i in range(num_plot):
        ax = plt.subplot(num_plot, 1, i+1)
        plt.plot(xdict['data'], ydict_list[i]['data'])
        plt.xlabel(xdict['label'])
        plt.ylabel(ydict_list[i]['label'])
        ax.legend([ydict_list[i]['name']], loc="best")
        if xdict['log']:
            ax.set_xscale('log')
        if ydict_list[i]['log']:
            ax.set_yscale('log')
    print("plot size=[" + str(size[0]) + "," + str(size[1]) + "]")
    fig.tight_layout()
    plt.show()
    return fig


class GenericWave(object):
    EFF_FREQ_DIGIT = 5
    EFF_TIME_DIGIT = 0 + 9

    def __init__(self):
        # fundamental attributes
        self._x
        self._y
        self._name
        pass

    @property
    def span(self):
        return round(
            self.x[-1] - self.x[0] + self.dx, self.__class__.EFF_TIME_DIGIT
            )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name=''):
        self._name = name

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def f(self):
        return fftfreq(len(self), self.dx)

    @property
    def yf(self):
        return fft(self.y)

    @property
    def dx(self):
        try:
            return round(
                self.x[1] - self.x[0], self.__class__.EFF_TIME_DIGIT
                )
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

    @property
    def faxis(self):
        return axis('t', 'time (s)', fftshift(self.f), False)

    @property
    def yfaxis(self):
        return axis('t', 'time (s)', fftshift(self.yf), False)

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

    def plot(self, figure_name='', plotDataOnly=False):
        """
        Plot y v.s. x.

        Returns
        -------
        matplotlib.lines.Line2D
            Plot handler object.

        """
        xdict = self.xaxis
        ydict_list = [{
            'name': self._name, 'label': 'amplitude',
            'data': self.y, 'log': False
            }]
        if plotDataOnly:
            return xdict, ydict_list, figure_name
        return draw(xdict, ydict_list, figure_name)

    def diff(self, n=1):
        """
        Calculate y n-th derivative using fft method.

        Parameters
        ----------
        n : int, optional
            order of differentiation. The default is 1.

        Returns
        -------
        numpy.array
            y derivative of n-th order.

        """
        return ifft((1j * 2 * np.pi * self.f)**n * fft(self.y)).real

    def psd(self, dBm_scale=True):
        """
        Calculate PSD using FFT.
        ref=https://stackoverflow.com/questions/20165193/fft-normalization

        #######################################################
        # FFT using Welch method
        # windows = np.ones(nfft) - no windowing
        # if windows = 'hamming', etc.. this function will
        # normalize to an equivalent noise bandwidth (ENBW)
        #######################################################

        Returns
        -------
        list
            FFT outputs, including frequency, lineaer scale psd.

        """
        nfft = len(self)  # fft size same as signal size
        df = 1 / self.dx
        f, Pxx_den = welch(
            self.y, fs=df, window=np.ones(nfft),
            nperseg=nfft, scaling='density'
            )
        if dBm_scale:
            return f, 10.0 * np.log10(Pxx_den)
        return f, Pxx_den

    def psdplot(self, dBm_scale=True, plotDataOnly=False):
        """
        Plot PSD of the signal.

        Parameters
        ----------
        dBm_scale : boolean, optional
            Show plot in dBm scale. The default is True.

        Returns
        -------
        handle : matplotlib.lines.Line2D
            Plot handler object.

        """
        f, PSD = self.psd(dBm_scale)
        xdict = axis('', 'frequency (Hz)', f, False)
        ydict_list = [axis(self._name, 'amplitude (Mag/Hz)', PSD, False)]
        if dBm_scale:
            ydict_list[0]['label'] = 'amplitude (dBm/Hz)'
        if plotDataOnly:
            return xdict, ydict_list, 'PSD'
        return draw(xdict, ydict_list, 'PSD')

    @classmethod
    def save(cls, *args):
        """
        Save GenericWave object to .wtobj files.

        Parameters
        ----------
        cls : GenericWave class
            GenericWave class.
        *args : GenericWave
            Object to be saved.

        """
        for i, gwObj in enumerate(args):
            if gwObj.name == '':
                gwObj.name = input(
                    'Empty name string for {i}th item, set object name:'
                    )
            with open(f'{gwObj.name}.wtobj', 'wb') as f:
                pickle.dump(gwObj, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, *args):
        """
        Load GenericWave object from .wtobj files.

        Parameters
        ----------
        cls : GenericWave class
            GenericWave class.
        *args : String
            Filename.

        Returns
        -------
        GenericWave
            Loaded objects.

        """
        objList = []
        for filename in args:
            with open(filename, 'rb') as f:
                objList += [pickle.load(f)]
        return *objList,


class GenericGate(object):

    def __init__(self, *qcObj):
        temp = deepcopy(qcObj)
        temp = temp[0].__class__.alignQubitChannels(*temp)
        self._qubitDict = {qcObj0._name: qcObj0 for qcObj0 in temp}
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

    def __matmul__(self, qbname):
        """
        Return QubitChannel object according to specified name.

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

    @classmethod
    def save(cls, *args):
        """
        Save GenericGate object to .gate files.

        Parameters
        ----------
        cls : GenericGate class
            GenericGate class.
        *args : GenericGate
            Object to be saved.

        """
        for i, gwObj in enumerate(args):
            if gwObj.name == '':
                gwObj.name = input(
                    'Empty name string for {i}th item, set object name:'
                    )
            with open(f'{gwObj.name}.gate', 'wb') as f:
                pickle.dump(gwObj, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, *args):
        """
        Load GenericGate object from .gate files.

        Parameters
        ----------
        cls : GenericGate class
            GenericGate class.
        *args : String
            Filename.

        Returns
        -------
        GenericGate
            Loaded objects.

        """
        objList = []
        for filename in args:
            with open(filename, 'rb') as f:
                objList += [pickle.load(f)]
        return *objList,
