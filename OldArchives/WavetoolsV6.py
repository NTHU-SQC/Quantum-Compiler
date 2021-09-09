# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 20:18:43 2021

for set / get vs IDE:
    https://stackoverflow.com/questions/52312897

@author: Alaster
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.signal import welch
from scipy.fft import fft, ifft, fftfreq, fftshift


class GenericWave():
    EFF_FREQ_DIGIT = 5
    EFF_TIME_DIGIT = 5 + 9

    def __init__(self):
        # fundamental attributes
        self._x
        self._y
        self._name
        pass

    @property
    def span(self):
        return self._x[-1] - self._x[0]

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
        return fft(self._y)

    @property
    def dx(self):
        try:
            return round(
                self._x[1] - self._x[0], self.__class__.EFF_TIME_DIGIT
                )
        except(IndexError):
            return 0

    @property
    def sample_rate(self):
        return round(1 / self.dx, self.__class__.EFF_FREQ_DIGIT)

    @property
    def xy(self):
        return self._x, self._y

    @property
    def xaxis(self):
        return self.__class__.axis('t', 'time (s)', self._x, False)

    @property
    def faxis(self):
        return self.__class__.axis('t', 'time (s)', fftshift(self.f), False)

    @property
    def yfaxis(self):
        return self.__class__.axis('t', 'time (s)', fftshift(self.yf), False)

    def __len__(self):
        """
        Get number of points in the object. Denoted as len(self).

        Returns
        -------
        int
            Number of points in the object.

        """
        return len(self._x)

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
        return np.array([np.interp(x, self._x, self._y) for x in xList])

    def plot(self, figure_name=''):
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
            'data': self._y, 'log': False
            }]
        return self.__class__.draw(xdict, ydict_list, figure_name)

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
        return ifft((1j * 2 * np.pi * self.f)**n * fft(self._y)).real

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
        sample_rate = 1 / self.dx
        f, Pxx_den = welch(
            self._y, fs=sample_rate, window=np.ones(nfft),
            nperseg=nfft, scaling='density'
            )
        if dBm_scale:
            return f, 10.0 * np.log10(Pxx_den)
        return f, Pxx_den

    def psdplot(self, dBm_scale=True):
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
        xdict = self.__class__.axis('', 'frequency (Hz)', f, False)
        ydict_list = [
            self.__class__.axis(self._name, 'amplitude (Mag/Hz)', PSD, False)
            ]
        if dBm_scale:
            ydict_list[0]['label'] = 'amplitude (dBm/Hz)'
        return self.__class__.draw(xdict, ydict_list, 'PSD')

    @classmethod
    def axis(cls, name='', label='', data=np.array([]), log_bool=False):
        return {'name': name, 'label': label, 'data': data, 'log': log_bool}

    @classmethod
    def draw(cls, xdict={}, ydict_list=[], figure_name='', titleFontSize=20,
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
            xdict = cls.axis(data=range(len(ydict_list[0]['data'])))
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


class Wave(GenericWave):

    def __init__(self, funcHandle=None, variables=[], properties={}):
        """
        Encapsulation of outputs generated by user defined function (shape
        function) with the capability to further modify wave's amplitude, shape
        and phase.

        Parameters
        ----------
        funcHandle : function object, optional
            wave shape generating function object, use None to enable
            dictionary-object creation of Wave object. The default is None.
        variables : list, optional
            list of variable to be passed to funcHandle. The default is [].
        properties : dict, optional
            Use dictionary-object to create Wave object, available only if
            funcHandle is None. The default is {}.

        Returns
        -------
        Wave
            An encapsulated Wave object.

        """
        if funcHandle is None:
            self.__properties = properties
        else:
            self.__properties = funcHandle(*variables)
        self._name = deepcopy(self.__properties['name'])
        self.variables = deepcopy(self.__properties['variables'])
        self._y = deepcopy(self.__properties['y'])
        self._x = deepcopy(self.__properties['x'])
        self._appendRule = deepcopy(self.__properties['appendRule'])

    @property
    def appendRule(self):
        """
        Get method for append rule settings.

        Returns
        -------
        list
            Append rule settings.

        """
        return self._appendRule

    @appendRule.setter
    def appendRule(self, appendRule):
        """
        Set method for append rule settings. This determines the appending
        behaviour among Waveform objects.

        Parameters
        ----------
        appendRule : list, optional
            List of booleans, first/last slot is for head/tail priority. Set
            True to prioritize the head/tail while False is for overlapping
            mode. The appending rule is as follows:
                wave(False) + wave(False) => Appending with overlapping while
                        keeping both the last point (fisrt wave) and the first
                        point (second wave) by averging them (by 2).
                wave(True) + wave(False) => Appending with overlapping while
                        keeping the last point (fisrt wave) and neglecting the
                        first point (second wave).
                wave(False) + wave(True) => Appending with overlapping while
                        neglecting the last point (fisrt wave) and keeping the
                        first point (second wave).
                wave(True) + wave(True) => Appending without overlapping while
                        both the last point (fisrt wave) and the first point
                        (second wave) are kept with an addition offset dx is
                        set between them.

        """
        self._appendRule = appendRule

    def __invert__(self):
        """
        Shorthand conversion for toWaveform() method. The overlapping mode is
        set to enabled. Denoted as ~self.

        Returns
        -------
        Waveform
             Object with a new reference.

        """
        return self.toWaveform()

    def __pos__(self):
        """
        Duplicate a Wave object with a new reference (also for each subentry).
        Denoted as +self.

        Returns
        -------
        Wave
             Object with a new reference.

        """
        return Wave(properties=self.__properties)

    def __neg__(self):
        """
        Reverse y array of a wave. Denoted as -self.

        Returns
        -------
        Wave
             Object with a new reference.

        """
        properties = {'name': self._name,
                      'variables': self.variables,
                      'y': self._y[::-1],
                      'x': self._x,
                      'appendRule': self._appendRule
                      }
        return Wave(properties=properties)

    def __add__(self, waveObj):
        """
        Superposition of waves by addition. Denoted as self + waveObj.

        Parameters
        ----------
        waveObj : Wave or float
            Operand.

        Returns
        -------
        Wave
            Object with a new reference.

        """
        if not isinstance(waveObj, Wave):
            properties = {'name': self._name,
                          'variables': self.variables,
                          'y': self._y + waveObj,
                          'x': self._x,
                          'appendRule': self._appendRule
                          }
            return Wave(properties=properties)
        variables = {
            'function': self.__add__, 'Augend': self, 'Addend': waveObj}
        longer = max(self, waveObj)
        if self is waveObj:
            shorter = self
        else:
            shorter = [obj for obj in [self, waveObj] if obj is not longer][0]
        x, length = longer._x, len(shorter)
        y = np.concatenate((
            longer._y[:length] + shorter._y[:length], longer._y[length:]
            ))
        properties = {'name': self._name,
                      'variables': variables,
                      'x': x,
                      'y': y,
                      'appendRule': [i or j for i, j in zip(
                                      self._appendRule, waveObj._appendRule
                                      )]
                      }
        return Wave(properties=properties)

    def __radd__(self, other):
        # Image method for __add__
        # see https://www.cnblogs.com/scolia/p/5686267.html
        return self + other

    def __sub__(self, waveObj):
        """
        Superposition of waves by substraction. Denoted as self - waveObj.

        Parameters
        ----------
        waveObj : Wave or float
            Operand.

        Returns
        -------
        Wave
            Object with a new reference.

        """
        if not isinstance(waveObj, Wave):
            properties = {'name': self._name,
                          'variables': self.variables,
                          'y': self._y - waveObj,
                          'x': self._x,
                          'appendRule': self._appendRule
                          }
            return Wave(properties=properties)
        variables = {
            'function': self.__sub__, 'Minuend': self, 'Subtrahend': waveObj}
        longer = max(self, waveObj)
        if self is waveObj:
            shorter = self
        else:
            shorter = [obj for obj in [self, waveObj] if obj is not longer][0]
        x, length = longer._x, len(shorter)
        y = np.concatenate((
            longer._y[:length] - shorter._y[:length], longer._y[length:]
            ))
        properties = {'name': self._name,
                      'variables': variables,
                      'x': x,
                      'y': y,
                      'appendRule': [i or j for i, j in zip(
                                      self._appendRule, waveObj._appendRule
                                      )]
                      }
        return Wave(properties=properties)

    def __rsub__(self, other):
        # Image method for __sub__
        # https://www.cnblogs.com/scolia/p/5686267.html
        return self - other

    def __mul__(self, waveObj):
        """
        Superposition of waves by multiplication. Denoted as self * waveObj.

        Parameters
        ----------
        waveObj : Wave or float
            Operand.

        Returns
        -------
        Wave
            Object with a new reference.

        """
        if not isinstance(waveObj, Wave):
            properties = {'name': self._name,
                          'variables': self.variables,
                          'y': self._y * waveObj,
                          'x': self._x,
                          'appendRule': self._appendRule
                          }
            return Wave(properties=properties)
        variables = {
            'function': self.__mul__, 'Multiplicand': self,
            'Multiplier': waveObj}
        longer = max(self, waveObj)
        if self is waveObj:
            shorter = self
        else:
            shorter = [obj for obj in [self, waveObj] if obj is not longer][0]
        x, length = longer._x, len(shorter)
        y = np.concatenate((
            longer._y[:length] * shorter._y[:length], longer._y[length:]
            ))
        properties = {'name': self._name,
                      'variables': variables,
                      'x': x,
                      'y': y,
                      'appendRule': [i or j for i, j in zip(
                                      self._appendRule, waveObj._appendRule
                                      )]
                      }
        return Wave(properties=properties)

    def __rmul__(self, other):
        # Image method for __mul__
        # https://www.cnblogs.com/scolia/p/5686267.html
        return self * other

    def __truediv__(self, number):
        """
        Wave amplitude division by a number. Denoted as self / number.

        Parameters
        ----------
        number : float
            Operand.

        Returns
        -------
        Wave
            Object with a new reference.

        """
        if not isinstance(number, Wave):
            properties = {'name': self._name,
                          'variables': self.variables,
                          'y': self._y / number,
                          'x': self._x,
                          'appendRule': self._appendRule
                          }
            return Wave(properties=properties)
        else:
            print('Divider must be numeric')
            return self

    def __rtruediv__(self, other):
        # Image method for __truediv__
        # https://www.cnblogs.com/scolia/p/5686267.html
        return self / other

    def __str__(self):
        """
        Print the status of a wave. Denoted as print(self).

        Returns
        -------
        string
            The status of a wave.

        """
        return f"name: {self.name}\n" + \
               f"variable: {self.variables}\n" + \
               f"dx: {self.dx}\n" + \
               f"appendRule: {self.appendRule}\n" + \
               f"ID: {id(self)}"

    def __abs__(self):
        """
        Taking absolute value of y of a Wave object. Denoted as abs(self).

        Returns
        -------
        Wave
            Object with a new reference.

        """
        properties = {'name': self._name,
                      'variables': self.variables,
                      'y': abs(self._y),
                      'x': self._x,
                      'appendRule': self._appendRule
                      }
        return Wave(properties=properties)

    def toWaveform(self):
        """
        Shorthand conversion to waveform object.

        Returns
        -------
        Waveform
            New waveform object.

        """
        return Waveform([self], self._name)


class Waveform(GenericWave):

    def __init__(self, waveObjList=[], name=''):
        """
        A interface to manipulate the order of waves.

        Parameters
        ----------
        waveObjList : list, optional
            Ordered list of Wave objects to be compiled into waveform (pulse
            train). The default is [].
        name : string, optional
            Name of waveform. The default is ''.

        Returns
        -------
        Waveform
            New waveform object.

        """
        self._waveList = deepcopy(waveObjList)
        self._name = deepcopy(name)
        self._y, self._x = self.__class__._synthesize(self._waveList)

    @property
    def waveList(self):
        """
        Get method for self._waveList, which contains the element waves of the
        waveform.

        Returns
        -------
        list
            A list of Wave objects.

        """
        return self._waveList

    @waveList.setter
    def waveList(self, waveObjList):
        """
        Set method for self._waveList, which contains the element waves of the
        waveform.

        Parameters
        ----------
        waveObjList : list
            A list of Wave objects to be used to setup the waveform.

        """
        if not waveObjList:
            raise ValueError("waveObjList cannot be empty")
        self._waveList = waveObjList
        self._y, self._x = self.__class__._synthesize(self._waveList)

    @property
    def appendRule(self):
        """
        Get method for appendRule of entire waveform.

        Returns
        -------
        list
            A list of appendRules.

        """
        return [
            self._waveList[0].appendRule[0], self._waveList[-1].appendRule[-1]
            ]

    def __str__(self):
        """
        Print the status of a waveform.

        Returns
        -------
        string
            The status of a waveform.

        """
        return f"name: {self.name}\n" + \
               f"wave list: {self.waveList}\n" + \
               f"dx: {self.dx}\n" + \
               f"ID: {id(self)}"

    def __pos__(self):
        """
        Duplicate a Waveform object with a new reference. Denoted as +self.

        Returns
        -------
        Waveform
             Object with a new reference.

        """
        return Waveform(self._waveList)

    def __invert__(self):
        """
        Shorthand conversion for toQubitChannel() method. Denoted as ~self.

        Returns
        -------
        QubitChannel
             Object with a new reference.

        """
        return self.toQubitChannel()

    def __add__(self, waveObjList):
        """
        Addition of 2 waveform objects. Denoted as self + waveObjList.

        Parameters
        ----------
        waveObjList : list or Waveform
            List of Wave objects or Waveform.

        Returns
        -------
        Waveform
            Object with a new reference.

        """
        waveObjList = self.__class__._toWaveObjList(waveObjList)
        waveList_new = self._waveList + waveObjList
        return Waveform(waveList_new)

    def __mul__(self, num):
        """
        Duplication of waveform object. Denoted as self * num.

        Parameters
        ----------
        num : int
            Number of duplications.

        Returns
        -------
        Waveform
            Object with a new reference.

        """
        waveList_new = self._waveList * num
        return Waveform(waveList_new)

    def __rmul__(self, other):
        # Image method for __mul__
        # https://www.cnblogs.com/scolia/p/5686267.html
        return self * other

    def __rshift__(self, offset):
        """
        Shorthand operator for self.offset() with add_tail=False and. Denoted
        as self >> waveform.

        Parameters
        ----------
        offset : float
            Offset value with the same unit as x.

        Returns
        -------
        Waveform
            Offsetted result with a new reference.

        """
        return self.offset(offset, add_tail=False)

    def __irshift__(self, waveform):
        """
        Shorthand operator for self.alignwith() with use_1st_head=False and
        align_2nd_head=False. Denoted as self >>= waveform.

        Parameters
        ----------
        waveform : Waveform
            Waveform to be aligned with.

        Returns
        -------
        None.

        """
        self.alignwith(waveform, use_1st_head=False, align_2nd_head=False)
        return self

    def __lshift__(self, offset):
        """
        Shorthand operator for self.offset() with add_tail=True and. Denoted as
        self << waveform.

        Parameters
        ----------
        offset : float
            Offset value with the same unit as x.

        Returns
        -------
        Waveform
            Offsetted result with a new reference.

        """
        return self.offset(offset, add_tail=True)

    def __ilshift__(self, waveform):
        """
        Shorthand operator for self.alignwith() with use_1st_head=True and
        align_2nd_head=True. Denoted as self <<= waveform.

        Parameters
        ----------
        waveform : Waveform
            Waveform to be aligned with.

        Returns
        -------
        None.

        """
        self.alignwith(waveform, use_1st_head=True, align_2nd_head=True)
        return self

    def permute(self, order):
        """
        Re-ordering the Wave object list of a waveform.

        Parameters
        ----------
        order : list
            List of indices for new positions.

        Returns
        -------
        None.

        """
        waveList_ref = self._waveList
        waveList = [waveList_ref[i] for i in order]
        self.waveList = waveList

    def remove(self, indices=[]):
        """
        Remove a wave from a waveform by specified indices.

        Parameters
        ----------
        indices : list, optional
            Indices to be removed. The default is [].

        Returns
        -------
        None.

        """
        waveList_ref = self._waveList
        waveList = [
            waveList_ref[index] for index in range(
                len(waveList_ref)) if index not in indices
                ]
        self.waveList = waveList

    def insert(self, indices=[], waveObjList=[]):
        """
        Insert a list of Wave objects to a waveform with specified indices.

        Parameters
        ----------
        indices : list, optional
            Indices to be inserted with waves. The default is [].
        waveObjList : list, optional
            List of Wave objects to be inserted. The default is [].

        Returns
        -------
        None.

        """
        waveList = self._waveList
        if len(waveObjList) > len(indices):
            indices = (indices[0] + np.array(range(len(waveObjList)))).tolist()
        for index, waveObj in zip(indices, waveObjList):
            waveList.insert(index, waveObj)
        self.waveList = waveList

    def replace(self, indices=[], waveObjList=[]):
        """
        Replace a list of Wave objects to a waveform with specified indices.

        Parameters
        ----------
        indices : list, optional
            Indices to be replaced with waves. The default is [].
        waveObjList : list, optional
            List of Wave objects to be replaced. The default is [].

        Returns
        -------
        None.

        """
        waveList = self._waveList
        if len(waveObjList) > len(indices):
            indices = (indices[0] + np.array(range(len(waveObjList)))).tolist()
        for index, waveObj in zip(indices, waveObjList):
            try:
                waveList[index] = waveObj
            except(IndexError):
                waveList += [waveObj]
        self.waveList = waveList

    def split(self, appendOverlap=True):
        """
        Split the waveform into sub-waveforms.

        Parameters
        ----------
        appendOverlap : boolean, optional
            True if the new sub-waveforms are in overlapping mode. The default
            is True.

        Returns
        -------
        list
            List of sub-waveforms.

        """
        return [~waveObj for waveObj in self._waveList]

    def alignwith(self, waveform, use_1st_head=True, align_2nd_head=True):
        """
        Perform algnment between a waveforms by appending 0s to 1 or 2 of the
        waveforms. Both waveforms are modified in the end, so no new reference
        is generated.

        Parameters
        ----------
        waveform : Waveform
            Waveform to be aligned with.
        use_1st_head : boolean, optional
            Corresponding to Self, as the 1st object. The default is True.
        align_2nd_head : boolean, optional
            Corresponding to waveform, as the 2nd object. The default is True.

        Returns
        -------
        None.

        """
        if waveform == self:
            return
        samp_rate = round(1 / self.dx, 5)
        if use_1st_head ^ align_2nd_head:
            addListA = self.__class__._nullBlock(
                self.span, samp_rate, self.appendRule)
            addListB = self.__class__._nullBlock(
                waveform.span, samp_rate, waveform.appendRule)
            if use_1st_head:
                self.waveList = addListB + self.waveList
                waveform.waveList = waveform.waveList + addListA
            else:
                self.waveList = self.waveList + addListB
                waveform.waveList = addListA + waveform.waveList
        else:
            span = round(abs(self.span - waveform.span), 9 + 10)
            # distinguish longer & shorter waveform
            longer = max(self, waveform)
            shorter = min(self, waveform)
            if use_1st_head:
                if not shorter.appendRule[-1]:
                    span += self.dx
                addList = self.__class__._nullBlock(
                    span, samp_rate, [True, longer.appendRule[-1]])
                shorter.waveList = shorter.waveList + addList
            else:
                if not shorter.appendRule[0]:
                    span += self.dx
                addList = self.__class__._nullBlock(
                    span, samp_rate, [longer.appendRule[0], True])
                shorter.waveList = addList + shorter.waveList

    def toQubitChannel(self):
        """
        Shorthand conversion to QubitChannel object.

        Returns
        -------
        QubitChannel
            New QubitChannel object.

        """
        return QubitChannel(self)

    def offset(self, offset=0.):
        """
        Backend function to offset the first or the last Wave object in a
        waveform.

        Parameters
        ----------
        offset : float, optional
            Offset by a value with the same unit as x. The default is 0.

        Returns
        -------
        None.

        """
        if abs(offset) < self.dx:
            return self
        span = offset
        samp_rate = round(1 / self.dx, 5)
        if offset > 0:
            if not self.appendRule[0]:
                span += self.dx
            return Waveform(
                self.__class__._nullBlock(span, samp_rate, [True, True]) +
                self.waveList
                )
        else:
            if not self.appendRule[-1]:
                span += self.dx
            return Waveform(
                self.waveList +
                self.__class__._nullBlock(span, samp_rate, [True, True])
                )

    @classmethod
    def _nullBlock(cls,
                   span=.0,
                   sampling_rate=1e9,
                   appendRule=[True, True],
                   delEnd=True):
        """
        Generate 0s to fill up empty space.

        Parameters
        ----------
        cls : Waveform class
            Waveform class object.
        span : float, optional
            Time span. The default is .0.
        sampling_rate : float, optional
            Sampling rate for DAC. The default is 1e9 (Suggested).
        appendRule : list, optional
            List of append rules. The default is [False, False].

        Returns
        -------
        list
            list for Waveform object generation.

        """
        points = int(span * sampling_rate + 1)
        variables = {'function': cls._nullBlock,
                     'sampling_rate': sampling_rate,
                     'span': span}
        end = None
        if delEnd:
            end = -1
        x = np.linspace(0, span, points)[:end]
        return [Wave(properties={'name': 'null',
                                 'variables': variables,
                                 'x': x,
                                 'y': np.zeros(x.size),
                                 'appendRule': appendRule})]

    @classmethod
    def _synthesize(cls, waveList):
        """
        Modded in V6
        Backend function to compile the list of Wave objects into a complete
        waveform (pulse train).

        Parameters
        ----------
        cls : Waveform class
            Waveform class object.
        waveList : list
            List of Wave objects to compiled into waveform.
        appendOverlap : boolean, optional
            Set overlapping mode. The default is False.

        Returns
        -------
        y : numpy.array
            y data.
        x : numpy.array
            x data.

        """
        y = np.array([])
        x = np.array([])
        offset = 0
        for waveObj in waveList:
            if x.size == 0:
                y = waveObj._y
                x = waveObj._x
                if x.size == 0:
                    continue
                try:
                    offset = x[-1]
                except(IndexError):
                    offset = 0
                previous = waveObj
                continue
            leftRule = previous.appendRule[1]
            rightRule = waveObj.appendRule[0]
            if leftRule ^ rightRule:
                if leftRule:
                    y = np.hstack([y, waveObj._y[1:]])
                else:
                    y = np.hstack([y[:-1], waveObj._y])
                x = np.hstack([x, waveObj._x[1:] + offset])
            else:
                if leftRule:
                    y = np.hstack([y, waveObj._y])
                    x = np.hstack([x, waveObj._x + offset + waveObj.dx])
                else:
                    y = np.hstack([y[:-1],
                                   np.array([(y[-1] + waveObj._y[0])/2]),
                                   waveObj._y[1:]])
                    x = np.hstack([x, waveObj._x[1:] + offset])
            offset = x[-1]
            previous = waveObj
        return y, np.round(x, cls.EFF_TIME_DIGIT)

    @classmethod
    def _toWaveObjList(cls, waveform):
        """
        Backend function to transform the input into a list of Wave objects.

        Parameters
        ----------
        cls : Waveform class
            Waveform class object.
        waveform : Waveform or Wave
            Waveform or Wave object.

        Returns
        -------
        list
            List of Wave objects.

        """
        if isinstance(waveform, Waveform):
            return waveform._waveList
        if isinstance(waveform, Wave):
            return [waveform]
        if isinstance(waveform, list):
            return waveform
        else:
            raise TypeError("Incorrect data type")


class QubitChannel(GenericWave):

    def __init__(self, *waveforms):
        """
        An interface for qubit-wise manipulation of waveform ordering. Each
        waveform is assigned to an independent wire and alignment is performed
        among wires.

        Parameters
        ----------
        *waveforms : Waveform
            Waveforms to be assigned to wires.

        Returns
        -------
        QubitChannel
            New QubitChannel object.

        """
        self._wires = [*waveforms]
        self._wire_names = [waveform._name for waveform in self._wires]
        self.__class__._align(self)
        self._x = self._wires[0]._x
        self._y = [self._wires[i]._y for i in range(len(self._wires))]
        self._name = ''

    @property
    def wire_names(self):
        """
        Get method for wire name strings.

        Returns
        -------
        list
            List of wire name strings.

        """
        return self._wire_names

    @wire_names.setter
    def wire_names(self, nameList=[]):
        """
        Set method for wire name strings.

        Parameters
        ----------
        nameList : list, optional
            Input list of name strings. The default is [].

        Returns
        -------
        None.

        """
        self._wire_names = nameList + [
                '' for i in range(len(self._wire_names) - len(nameList))]

    def __str__(self):
        """
        Print the status of a QubitChannel object.

        Returns
        -------
        string
            The status of a QubitChannel object.

        """
        return f"name: {self.name}\n" + \
            f"wires: {self._wires}\n" + \
            f"wire names: {self._wire_names}\n" + \
            f"dx: {self.dx}\n" + \
            f"ID: {id(self)}"

    def __add__(self, qcObj):
        """
        Addition of 2 QubitChannel objects. Denoted as self + qcObj.

        Parameters
        ----------
        qcObj : QubitChannel
            QubitChannel object.

        Returns
        -------
        QubitChannel
            Appended QubitChannel object with a new reference.

        """
        if len(self._wires) != len(qcObj._wires):
            raise ValueError("wire number mismatch")
        wires = [wireA + wireB for wireA, wireB in zip(
                self._wires, qcObj._wires)]
        return QubitChannel(*wires)

    def __mul__(self, num):
        """
        Duplication of a QubitChannel object. Denoted as self * num.

        Parameters
        ----------
        num : int
            Number of duplications.

        Returns
        -------
        Gate
            Appended QubitChannel object with a new reference.

        """
        wires = [wire * num for wire in self._wires]
        return QubitChannel(*wires)

    def __rmul__(self, other):
        # Image method for __mul__
        # https://www.cnblogs.com/scolia/p/5686267.html
        return self * other

    def __truediv__(self, qcObj):
        """
        Shorthand method for add_wire() method. Denoted as self / qcObj.

        Parameters
        ----------
        qcObj : list or QubitChannel or Waveform
            List of Waveforms (or a Waveform object) or a QubitChannel object.

        Returns
        -------
        Wave
            Modified QubitChannel object with a new reference.

        """
        return self.add_wire(qcObj)

    def __matmul__(self, xList):
        """
        Return each wire y values at x in xList.

        Parameters
        ----------
        xList : numpy.array or list.
            Array or list of x values.

        Returns
        -------
        list
            List of corresponding y values.

        """
        return [self._wires[i]@(xList) for i in range(len(self._wires))]

    def add_wire(self, waveformList=[]):
        """
        Add additional wires to QubitChannel object.

        Parameters
        ----------
        waveformList : llist or QubitChannel or Waveform
            List of Waveforms (or a Waveform object) or a QubitChannel object.

        Returns
        -------
        QubitChannel
            Appended QubitChannel object with a new reference.

        """
        waveformList = self.__class__._toWaveformObjList(waveformList)
        wires = self._wires + waveformList
        return QubitChannel(*wires)

    def plot(self, wire_indices=[], size=[6.4, 4.8], figure_name=''):
        """
        A quick plot among wires of a QubitChannel object.

        Parameters
        ----------
        wire_indices : list, optional
            List of indices of wires to be examined. The default is [].
        size : list, optional
            Size of each subplot. The default is [6.4, 4.8].

        Returns
        -------
        axisObjArr : list
            List of 2D line objects.

        """
        if not wire_indices:
            wire_indices = range(len(self._wires))
        ydict_list = [{}] * len(wire_indices)
        for idx, i in zip(wire_indices, range(len(wire_indices))):
            ydict_list[i] = self.__class__.axis(
                self._wire_names[idx], 'amplitude', self._y[idx], False
                )
        if not figure_name:
            figure_name = self._name
        return self.__class__.draw(
            self.xaxis, ydict_list, figure_name=figure_name, size=size
            )

    def psdplot(self, wire_indices=[], size=[6.4, 4.8], dBm_scale=True):
        """
        Plot PSD of the signal.

        Parameters
        ----------
        wire_indices : list, optional
            List of indices of wires to be examined. The default is [].
        size : list, optional
            Size of each subplot. The default is [6.4, 4.8].
        dBm_scale : boolean, optional
            Show plot in dBm scale. The default is True.

        Returns
        -------
        handle : matplotlib.lines.Line2D
            Plot handler object.

        """
        if not wire_indices:
            wire_indices = range(len(self._wires))
        ydict_list = [{}] * len(wire_indices)
        for idx, i in zip(wire_indices, range(len(wire_indices))):
            f, PSD = self._wires[idx].psd(dBm_scale)
            xdict = {
                'name': '', 'label': 'frequency (Hz)',
                'data': f, 'log': False
                }
            ydict_list[i] = {
                'name': self._wire_names[idx], 'label': 'amplitude (Mag/Hz)',
                'data': PSD, 'log': False
                }
            if dBm_scale:
                ydict_list[i]['label'] = 'amplitude (dBm/Hz)'
        return self.__class__.draw(xdict, ydict_list, 'PSD', size=size)

    @classmethod
    def _align(cls, qcObj):
        """
        Backend method to perform wire-wise alignment.

        Parameters
        ----------
        cls : QubitChannel class
            QubitChannel class.
        qcObj : QubitChannel
            QubitChannel object with wires to be aligned.

        Returns
        -------
        None.

        """
        longest = max(qcObj._wires, key=len)
        for waveform in qcObj._wires:
            if waveform is not longest:
                longest <<= waveform

    @classmethod
    def _toWaveformObjList(cls, qcObj):
        """
        Backend function to transform the input into a list of waveform
        objects.

        Parameters
        ----------
        cls : QubitChannel class
            QubitChannel class object.
        qcObj : list or QubitChannel or Waveform
            List of Waveforms (or a Waveform object) or a QubitChannel object.

        Returns
        -------
        list
            List of Waveform objects.

        """
        if isinstance(qcObj, QubitChannel):
            return qcObj._wires
        if isinstance(qcObj, Waveform):
            return [qcObj]
        if isinstance(qcObj, list):
            return qcObj
        else:
            raise TypeError("Incorrect data type")


if __name__ == '__main__':
    import shape_functionV4 as sf
    marker = ~Wave(sf.square, [sf.get_x(10e-6), 0, 2e-6])
    a = marker.offset(3e-6)
    a.plot()
    pass
