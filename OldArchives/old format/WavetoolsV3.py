# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 20:18:43 2021

@author: Alaster
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from scipy import stats
import scipy.signal


class GenericWave():

    def __init__(self):
        # fundamental attributes
        self.x
        self.dx
        self.y
        self.name
        pass

    # in-built methods
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

    # set and get methods
    def get_span(self, show=False):
        """
        Return object x-axis span.

        Parameters
        ----------
        show : boolean, optional
            Toogle True to show on terminal. The default is True.

        Returns
        -------
        float
            The span of x.

        """
        if show:
            print(self.x[-1])
        return self.x[-1]

    def set_name(self, name=''):
        """
        Set method for name string.

        Parameters
        ----------
        name : string, optional
            Name of the object. The default is ''.

        Returns
        -------
        None.

        """
        self.name = name

    def get_name(self, show=False):
        """
        Get method for name string.

        Parameters
        ----------
        show : boolean, optional
            Toogle True to show on terminal. The default is True.

        Returns
        -------
        string
            Name of the object.

        """
        if show:
            print(self.name)
        return self.name

    def get_x(self):
        """
        Get method for x data.

        Returns
        -------
        numpy.array
            x data.

        """
        return self.x

    def get_y(self):
        """
        Get method for y data.

        Returns
        -------
        numpy.array
            y data.

        """
        return self.y

    def get_waveform(self):
        """
        Get method for x & y data.

        Returns
        -------
        numpy.array
            x data.
        numpy.array
            y data.

        """
        return self.x, self.y

    # algeberic methods for size comparison
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
        return np.array([np.interp(x, *self.get_waveform()) for x in xList])

    # auxiliary methods
    def plot(self):
        """
        Plot y v.s. x for a quick view

        Returns
        -------
        matplotlib.lines.Line2D
            Plot handler object.

        """
        return plt.plot(self.x, self.y)

    def cycdiff(self, order=1, ratio=100):
        """
        Perform y derivation with periodic boundary conditions.

        Parameters
        ----------
        order : int, optional
            Order of derivative. The default is 1.
        ratio : float, optional
            Range of filtering (number of std values). The default is 100.

        Returns
        -------
        numpy.array
            Derivative.

        """
        arr = self.get_y()
        derivative = np.diff(arr, order, append=arr[0: order]) / self.dx
        mode = stats.mode(derivative)[0][0]
        std = np.std(derivative)
        return np.clip(derivative, mode-std*ratio, mode+std*ratio)

    def fft(self):
        """
        Perform FFT.
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
            FFT outputs, including: [frequency, lineaer scale, dbm scale].

        """
        nfft = len(self)  # fft size same as signal size
        sample_rate = 1 / self.dx
        f, Pxx_den = scipy.signal.welch(
            self.get_y(), fs=sample_rate, window=np.ones(nfft),
            nperseg=nfft, scaling='density'
            )
        self.spectrum = [f, Pxx_den, 10.0*np.log10(Pxx_den)]
        return self.spectrum

    def fftplot(self, dBm_scale=True):
        """
        Plot FFT of the signal. Perform FFT if there no FFT done before.

        Parameters
        ----------
        dBm_scale : boolean, optional
            Show plot in dBm scale. The default is True.

        Returns
        -------
        handle : matplotlib.lines.Line2D
            Plot handler object.

        """
        plt.figure()
        if not hasattr(self, 'spectrum'):
            self.fft()
        handle = plt.plot(self.spectrum[0],
                          self.spectrum[2] if dBm_scale else self.spectrum[1]
                          )
        plt.xlabel('Freq (Hz)')
        plt.ylabel('dBm/Hz')
        # plt.ylim([-200, 0])
        plt.show()
        print("Peak at " + str(
            self.spectrum[0][np.argmax(self.spectrum[2])]) + ' Hz')
        return handle


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
        self.name = copy(self.__properties['name'])
        self.variables = copy(self.__properties['variables'])
        self.y = copy(self.__properties['y'])
        self.x = copy(self.__properties['x'])
        try:
            self.dx = self.x[1] - self.x[0]
        except(IndexError):
            self.dx = 0
        self.appendRule = copy(self.__properties['appendRule'])

    # set and get methods
    def set_appendRule(self, appendRule=[False, False]):
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
            The default is [False, False].

        Returns
        -------
        None.

        """
        self.appendRule = appendRule

    def get_appendRule(self, show=False):
        """
        Get method for append rule settings.

        Parameters
        ----------
        show : boolean, optional
            Toogle True to show on terminal. The default is True.

        Returns
        -------
        list
            Append rule settings.

        """
        if show:
            print(self.appendRule)
        return self.appendRule

    # basic algeberic methods
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
        properties = {'name': self.name,
                      'variables': self.variables,
                      'y': self.y[::-1],
                      'x': self.x,
                      'appendRule': self.appendRule
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
            properties = {'name': self.name,
                          'variables': self.variables,
                          'y': self.y + waveObj,
                          'x': self.x,
                          'appendRule': self.appendRule
                          }
            return Wave(properties=properties)
        variables = {
            'function': self.__add__, 'Augend': self, 'Addend': waveObj}
        longer = max(self, waveObj)
        if self is waveObj:
            shorter = self
        else:
            shorter = [obj for obj in [self, waveObj] if obj is not longer][0]
        x, length = longer.x, len(shorter.x)
        y = np.concatenate((
            longer.y[:length] + shorter.y[:length], longer.y[length:]
            ))
        properties = {'name': self.name,
                      'variables': variables,
                      'x': x,
                      'y': y,
                      'appendRule': [i or j for i, j in zip(
                                      self.appendRule, waveObj.appendRule
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
            properties = {'name': self.name,
                          'variables': self.variables,
                          'y': self.y - waveObj,
                          'x': self.x,
                          'appendRule': self.appendRule
                          }
            return Wave(properties=properties)
        variables = {
            'function': self.__sub__, 'Minuend': self, 'Subtrahend': waveObj}
        longer = max(self, waveObj)
        if self is waveObj:
            shorter = self
        else:
            shorter = [obj for obj in [self, waveObj] if obj is not longer][0]
        x, length = longer.x, len(shorter.x)
        y = np.concatenate((
            longer.y[:length] - shorter.y[:length], longer.y[length:]
            ))
        properties = {'name': self.name,
                      'variables': variables,
                      'x': x,
                      'y': y,
                      'appendRule': [i or j for i, j in zip(
                                      self.appendRule, waveObj.appendRule
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
            properties = {'name': self.name,
                          'variables': self.variables,
                          'y': self.y * waveObj,
                          'x': self.x,
                          'appendRule': self.appendRule
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
        x, length = longer.x, len(shorter.x)
        y = np.concatenate((
            longer.y[:length] * shorter.y[:length], longer.y[length:]
            ))
        properties = {'name': self.name,
                      'variables': variables,
                      'x': x,
                      'y': y,
                      'appendRule': [i or j for i, j in zip(
                                      self.appendRule, waveObj.appendRule
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
            properties = {'name': self.name,
                          'variables': self.variables,
                          'y': self.y / number,
                          'x': self.x,
                          'appendRule': self.appendRule
                          }
            return Wave(properties=properties)
        else:
            print('Divider must be numeric')
            return self

    def __rtruediv__(self, other):
        # Image method for __truediv__
        # https://www.cnblogs.com/scolia/p/5686267.html
        return self / other

    # in-built methods
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
        properties = {'name': self.name,
                      'variables': self.variables,
                      'y': abs(self.y),
                      'x': self.x,
                      'appendRule': self.appendRule
                      }
        return Wave(properties=properties)

    # auxiliary methods
    def toWaveform(self):
        """
        Shorthand conversion to waveform object.

        Returns
        -------
        Waveform
            New waveform object.

        """
        return Waveform([self], self.name)


class Waveform(GenericWave):

    delEnd = True

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
        self.__waveList = copy(waveObjList)
        self.name = copy(name)
        self.y, self.x = Waveform._synthesize(self.__waveList)
        try:
            self.dx = self.x[1] - self.x[0]
        except(IndexError):
            self.dx = 0

    # set and get methods
    def set_waveList(self, waveObjList):
        """
        Set method for self.__waveList, which contains the element waves of the
        waveform.

        Parameters
        ----------
        waveObjList : list
            A list of Wave objects to be used to setup the waveform.

        Returns
        -------
        None.

        """
        if not waveObjList:
            print('Error: waveObjList cannot be empty')
            return
        self.__waveList = waveObjList
        self.y, self.x = Waveform._synthesize(self.get_waveList())

    def get_waveList(self):
        """
        Get method for self.__waveList, which contains the element waves of the
        waveform.

        Returns
        -------
        list
            A list of Wave objects.

        """
        return self.__waveList

    # in-built methods
    def __contains__(self, waveObj):
        """
        Check if waveObj is an element of the waveform.

        Parameters
        ----------
        waveObj : Wave
            Operand.

        Returns
        -------
        list
             Boolean list in which the waveObj location is specified by True.

        """
        return [obj is waveObj for obj in self.get_waveList()]

    def __str__(self):
        """
        Print the status of a waveform.

        Returns
        -------
        string
            The status of a waveform.

        """
        return f"name: {self.name}\n" + \
               f"wave list: {self.get_waveList()}\n" + \
               f"dx: {self.dx}\n" + \
               f"ID: {id(self)}"

    # algeberic methods
    def __pos__(self):
        """
        Duplicate a Waveform object with a new reference. Denoted as +self.

        Returns
        -------
        Waveform
             Object with a new reference.

        """
        return Waveform(self.get_waveList())

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
        waveObjList = Waveform._toWaveObjList(waveObjList)
        waveList_new = self.get_waveList() + waveObjList
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
        waveList_new = self.get_waveList() * num
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

    # index methods
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
        waveList_ref = self.get_waveList()
        waveList = [waveList_ref[i] for i in order]
        self.set_waveList(waveList)

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
        waveList_ref = self.get_waveList()
        waveList = [
            waveList_ref[index] for index in range(
                len(waveList_ref)) if index not in indices
                ]
        self.set_waveList(waveList)

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
        waveList = self.get_waveList()
        if len(waveObjList) > len(indices):
            indices = (indices[0] + np.array(range(len(waveObjList)))).tolist()
        for index, waveObj in zip(indices, waveObjList):
            waveList.insert(index, waveObj)
        self.set_waveList(waveList)

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
        waveList = self.get_waveList()
        if len(waveObjList) > len(indices):
            indices = (indices[0] + np.array(range(len(waveObjList)))).tolist()
        for index, waveObj in zip(indices, waveObjList):
            try:
                waveList[index] = waveObj
            except(IndexError):
                waveList += [waveObj]
        self.set_waveList(waveList)

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
        return [~waveObj for waveObj in self.get_waveList()]

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
        x, _ = [
            self + waveform if not (use_1st_head ^ align_2nd_head)
            else max(self, waveform)
            ][0].get_waveform()
        # distinguish longer & shorter waveform
        longer = max(self, waveform)
        shorter = [obj for obj in [self, waveform] if obj is not longer][0]
        # use 1st/last element to offset
        len_longer = longer.get_span(False)
        len_shorter = shorter.get_span(False)
        appendLength = abs(len_longer - len_shorter) if not (
            use_1st_head ^ align_2nd_head) else len_longer
        # modify each length
        if use_1st_head ^ align_2nd_head:
            longer_newList = longer.offset(
                len_shorter, shorter is self if use_1st_head
                else longer is self).get_waveList()
            shorter_newList = shorter.offset(
                    appendLength, longer is self if use_1st_head
                    else shorter is self).get_waveList()
            longer.set_waveList(longer_newList)
        else:
            shorter_newList = shorter.offset(
                appendLength, use_1st_head).get_waveList()
        shorter.set_waveList(shorter_newList)

    def toQubitChannel(self):
        """
        Shorthand conversion to QubitChannel object.

        Returns
        -------
        QubitChannel
            New QubitChannel object.

        """
        return QubitChannel(self)

    def offset(self, offset=0., add_tail=False):
        """
        Backend function to offset the first or the last Wave object in a
        waveform.

        Parameters
        ----------
        offset : float, optional
            Offset by a value with the same unit as x. The default is 0.
        add_tail : boolean, optional
            True to append 0s at tail, at head otherwise. The default is False.

        Returns
        -------
        None.

        """
        points = int(offset / self.dx+1)
        if points < 1:
            return self
        if add_tail:
            appendRule = [
                self.get_waveList()[-1].get_appendRule(False)[-1], True]
        else:
            appendRule = [True,
                          self.get_waveList()[0].get_appendRule(False)[0]]
        new_waveList = (
            self.get_waveList() + Waveform._nullBlock(
                offset, points, appendRule) if add_tail else
            Waveform._nullBlock(
                offset, points, appendRule) + self.get_waveList()
            )
        return Waveform(new_waveList)

    @classmethod
    def _nullBlock(
            cls, span=.0, points=0, appendRule=[False, False]):
        """
        Generate 0s to fill up empty space.

        Parameters
        ----------
        cls : Waveform class
            Waveform class object.
        span : float, optional
            Time span. The default is .0.
        points : int, optional
            Number of points. The default is 0.
        appendRule : list, optional
            List of append rules. The default is [False, False].

        Returns
        -------
        list
            list for Waveform object generation.

        """
        variables = {'function': cls._nullBlock,
                     'span': span,
                     'points': points}
        end, fix = None, 0
        if cls.delEnd:
            end = -1
            fix = 1
        return [Wave(properties={'name': 'null',
                                 'variables': variables,
                                 'x': np.linspace(0, span, points)[:end],
                                 'y': np.zeros(points-fix),
                                 'appendRule': appendRule})]

    @classmethod
    def _synthesize(cls, waveList):
        """
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
                y = waveObj.get_y()
                x = waveObj.get_x()
                if x.size == 0:
                    continue
                try:
                    offset = x[-1]
                    dx = x[1] - x[0]
                except(IndexError):
                    dx = 0
                previous = waveObj
                continue
            leftRule = previous.get_appendRule(False)[1]
            rightRule = waveObj.get_appendRule(False)[0]
            if leftRule ^ rightRule:
                if leftRule:
                    y = np.hstack([y, waveObj.y[1:]])
                else:
                    y = np.hstack([y[:-1], waveObj.y])
                x = np.hstack([x, waveObj.x[1:] + offset])
            else:
                if leftRule:
                    y = np.hstack([y, waveObj.y])
                    x = np.hstack([x, waveObj.x + offset + dx])
                else:
                    y = np.hstack([y[:-1],
                                   np.array([(y[-1] + waveObj.y[0])/2]),
                                   waveObj.y[1:]])
                    x = np.hstack([x, waveObj.x[1:] + offset])
            offset = x[-1]
            previous = waveObj
        return y, x

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
            return waveform.get_waveList()
        if isinstance(waveform, Wave):
            return [waveform]
        else:
            return waveform

    @classmethod
    def set_delEnd(cls, toggle=True):
        """
        Set method for delEnd attribute.

        Parameters
        ----------
        cls : Waveform class
            Waveform class object.
        toggle : boolean, optional
            Set True to delete tail index for all _nullBlock genertions. The
            default is True.

        Returns
        -------
        None.

        """
        cls.delEnd = toggle

    @classmethod
    def get_delEnd(cls, show=False):
        """
        Get method for delEnd attribute.

        Parameters
        ----------
        cls : Waveform class
            Waveform class object.
        show : boolean, optional
            Toogle True to show on terminal. The default is False.

        Returns
        -------
        boolean
            True if delEnd mode is on.

        """
        if show:
            print(cls.delEnd)
        return cls.delEnd


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
        self.wires = [*waveforms]
        self.number_of_wires = len(self.wires)
        self.wire_names = [waveform.get_name(False) for waveform in self.wires]
        QubitChannel._align(self)
        self.x = self.wires[0].get_x()
        try:
            self.dx = self.x[1] - self.x[0]
        except(IndexError):
            self.dx = 0
        self.y = [self.wires[i].get_y() for i in range(self.number_of_wires)]
        self.name = ''

    # get and Set methods
    def set_wire_names(self, nameList=[]):
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

        self.wire_names = nameList + [
                '' for i in range(len(self.wire_names) - len(nameList))]

    def get_wire_names(self, show=False):
        """
        Get method for wire name strings.

        Returns
        -------
        list
            List of wire name strings.

        """
        if show:
            print(self.wire_names)
        return self.wire_names

    # in-built methods
    def __str__(self):
        """
        Print the status of a QubitChannel object.

        Returns
        -------
        string
            The status of a QubitChannel object.

        """
        return f"name: {self.name}\n" + \
            f"wires: {self.wires}\n" + \
            f"wire names: {self.wire_names}\n" + \
            f"dx: {self.dx}\n" + \
            f"ID: {id(self)}"

    # algeberic methods
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
        if self.number_of_wires != qcObj.number_of_wires:
            print("Mismatch wire number!")
            return None
        wires = [wireA + wireB for wireA, wireB in zip(
                self.wires, qcObj.wires)]
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
        wires = [wire * num for wire in self.wires]
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
        return [self.wires[i]@(xList) for i in range(self.number_of_wires)]

    # auxiliary methods
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
        waveformList = QubitChannel._toWaveformObjList(waveformList)
        wires = self.wires + waveformList
        return QubitChannel(*wires)

    def plot(self, wire_indices=[], size=[6.4, 4.8]):
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
            wire_indices = range(self.number_of_wires)
        num_plot = len(wire_indices)
        axisObjArr, nameList = [], []
        fig = plt.figure(figsize=[size[0], size[1]*num_plot])
        fig.suptitle(self.get_name(False), fontsize=16)
        for index, i in zip(wire_indices, range(num_plot)):
            axisObjArr += [plt.subplot(num_plot, 1, i+1)]
            self.wires[i].plot()
            nameList += [self.wire_names[index]]
            axisObjArr[i].legend([nameList[i]], loc="best")
        print("plot size=["+str(size[0])+","+str(size[1])+"]")
        return axisObjArr

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
        longest = max(qcObj.wires, key=len)
        for waveform in qcObj.wires:
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
            return qcObj.wires
        if isinstance(qcObj, Waveform):
            return [qcObj]
        else:
            return qcObj
