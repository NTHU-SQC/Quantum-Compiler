# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:05:31 2021

@author: Alaster


1.example:
# a = Wave(gauss, [sigma, x, 50])
# b = Waveform([a])

"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy


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
        Get number of points in the Wave/Waveform object. Denoted as len(self).

        Returns
        -------
        int
            Number of points in the Wave/Waveform object.

        """
        return len(self.x)

    def __max__(self, *waveform):
        """
        Return wave/waveforms contains the largest number of points. Denoted
        as max(self, *waveform).

        Parameters
        ----------
        *waveform : Waveform or/and Wave objects
            Multiple waves or/and waveforms to be compared.

        Returns
        -------
        Waveform or Wave object
            Largest object.

        """
        return max(self, *waveform, key=len)

    def __min__(self, *waveform):
        """
        Return wave/waveforms contains the smallest number of points. Denoted
        as min(self, *waveform).

        Parameters
        ----------
        *waveform : Waveform or/and Wave objects
            Multiple waves or/and waveforms to be compared.

        Returns
        -------
        Waveform or Wave object
            Smallest object.

        """
        return min(self, *waveform, key=len)

    # set and get methods
    def get_span(self, show=False):
        """
        Return wave/waveforms x-axis span.

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
            Name of wave/waveform. The default is ''.

        Returns
        -------
        None.

        """
        self.name = name

    def get_name(self, show=False):
        """
        Get method for name string.

        Returns
        -------
        string
            Name of wave/waveform.

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
    def __lt__(self, waveform):
        """
        Operator method for self < waveform.

        Parameters
        ----------
        waveform : Wave or Waveform
            Operand.

        Returns
        -------
        Boolean
            True if len(self) < len(waveform).

        """
        return len(self) < len(waveform)

    def __le__(self, waveform):
        """
        Operator method for self <= waveform.

        Parameters
        ----------
        waveform : Wave or Waveform
            Operand.

        Returns
        -------
        Boolean
            True if len(self) <= len(waveform).

        """
        return len(self) <= len(waveform)

    def __eq__(self, waveform):
        """
        Operator method for self == waveform.

        Parameters
        ----------
        waveform : Wave or Waveform
            Operand.

        Returns
        -------
        Boolean
            True if len(self) == len(waveform).

        """
        return len(self) == len(waveform)

    def __ne__(self, waveform):
        """
        Operator method for self != waveform.

        Parameters
        ----------
        waveform : Wave or Waveform
            Operand.

        Returns
        -------
        Boolean
            True if len(self) != len(waveform).

        """
        return len(self) != len(waveform)

    def __ge__(self, waveform):
        """
        Operator method for self >= waveform.

        Parameters
        ----------
        waveform : Wave or Waveform
            Operand.

        Returns
        -------
        Boolean
            True if len(self) >= len(waveform).

        """
        return len(self) >= len(waveform)

    def __gt__(self, waveform):
        """
        Operator method for self > waveform.

        Parameters
        ----------
        waveform : Wave or Waveform
            Operand.

        Returns
        -------
        Boolean
            True if len(self) > len(waveform).

        """
        return len(self) > len(waveform)

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
            An encapsulated wave object.

        """
        if funcHandle is None:
            self.properties = properties
        else:
            self.properties = funcHandle(*variables)
        self.name = self.properties['name']
        self.variables = self.properties['variables']
        self.y = copy(self.properties['y'])
        self.x = copy(self.properties['x'])
        self.dx = self.x[1] - self.x[0]

    # set and get methods
    def add_offset(self, offset=.0, add_tail=False, reset=False):
        """
        Add an offset to the wave and return a composite wave.

        Parameters
        ----------
        offset : float, optional
            Offset to the head/tail. The unit is the same as x-axis. The
            default is .0.
        add_tail : boolean, optional
            Set True to append 0s to the end of the wave. The default is False.
        reset : boolean, optional
            Remove the offset option. The default is False.

        Returns
        -------
        Wave
            Modified wave object with a new reference.

        """
        if reset:
            try:
                return self.properties['variables']['wave']
            except(KeyError):
                return self
        pnts = int(abs(offset/(self.x[1]-self.x[0])))
        if offset >= 0:
            x = np.concatenate((
                self.x, np.linspace(
                    2 * self.x[-1] - self.x[-2], self.x[-1] + offset, pnts
                    )
                ))
            y = np.concatenate(
                (self.y, np.zeros(pnts))) if add_tail else np.concatenate(
                    (np.zeros(pnts), self.y))
        else:
            x = self.x[:len(self.x)-pnts]
            y = self.y[:len(self.x)-pnts] if add_tail else self.y[pnts:]
        variables = {'function': self.add_offset, 'offset': offset,
                     'wave': self, 'add_tail': add_tail, 'reset': reset}
        properties = {'name': '', 'variables': variables, 'x': x, 'y': y}
        return Wave(properties=properties)

    def get_function(self, show=True):
        """
        Get method for shape function object in a non-composite wave.

        Parameters
        ----------
        show : boolean, optional
            Print the name of the function. The default is True.

        Returns
        -------
        function object or None
            The shape function of the wave. None if the wave is composite.

        """
        if self.iscomposite():
            return None
        if show:
            print(self.variables['function'].__name__)
        return self.variables['function']

    # in-built methods
    def __str__(self):
        """
        Print the status of a wave.

        Returns
        -------
        string
            The status of a wave.

        """
        return f"name: {self.name}\n" + \
               f"variable: {self.variables}\n" + \
               f"dx: {self.dx}\n" + \
               f"composite: {self.iscomposite()}\n" + \
               f"ID: {id(self)}"

    # algeberic methods
    def __neg__(self):
        """
        Multiply wave object by -1. Denoted as -self.

        Returns
        -------
        Wave
            Modified wave object with a new reference.

        """
        return self * (-1)

    def __pos__(self):
        """
        Duplicate a wave object with a new reference, for each subentries the
        reference is the same but not for x & y. This is effectively self * 1.
        Denoted as ~self.

        Returns
        -------
        Wave
             wave object with a new reference.

        """
        return Wave(properties=self.properties)

    def __abs__(self):
        """
        Taking absolute value of y of a wave object. Denoted as abs(self).

        Returns
        -------
        Wave
            Modified wave object with a new reference.

        """
        properties = {'name': self.properties['name'],
                      'variables': self.properties['variables'],
                      'y': abs(self.properties['y']),
                      'x': self.properties['x']
                      }
        return Wave(properties=properties)

    def __invert__(self):
        """
        Shorthand conversion for toWaveform() method. The overlapping mode is
        set to enabled. Denoted as ~self.

        Returns
        -------
        Waveform
             New waveform object with a new reference.

        """
        return self.toWaveform(appendOverlap=True)

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
            Modified wave object with a new reference.

        """
        if not isinstance(waveObj, Wave):
            properties = {'name': self.properties['name'],
                          'variables': self.properties['variables'],
                          'y': self.properties['y'] + waveObj,
                          'x': self.properties['x']
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
        properties = {'name': '', 'variables': variables, 'x': x, 'y': y}
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
            Modified wave object with a new reference.

        """
        if not isinstance(waveObj, Wave):
            properties = {'name': self.properties['name'],
                          'variables': self.properties['variables'],
                          'y': self.properties['y'] - waveObj,
                          'x': self.properties['x']
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
        properties = {'name': '', 'variables': variables, 'x': x, 'y': y}
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
            Modified wave object with a new reference.

        """
        if not isinstance(waveObj, Wave):
            properties = {'name': self.properties['name'],
                          'variables': self.properties['variables'],
                          'y': self.properties['y'] * waveObj,
                          'x': self.properties['x']
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
        properties = {'name': '', 'variables': variables, 'x': x, 'y': y}
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
            Modified wave object with a new reference.

        """
        if not isinstance(number, Wave):
            properties = {'name': self.properties['name'],
                          'variables': self.properties['variables'],
                          'y': self.properties['y'] / number,
                          'x': self.properties['x']
                          }
            return Wave(properties=properties)
        else:
            print('Divider must be numeric')
            return self

    def __rtruediv__(self, other):
        # Image method for __truediv__
        # https://www.cnblogs.com/scolia/p/5686267.html
        return self / other

    def __rshift__(self, offset):
        """
        Shorthand operator for self.add_offset() with add_tail=False. Denoted
        as self >> offset.

        Parameters
        ----------
        offset : float
            Offset value with the same unit as x.

        Returns
        -------
        Wave
            Offsetted result with a new reference.

        """
        return self.add_offset(offset, add_tail=False)

    def __lshift__(self, offset):
        """
        Shorthand operator for self.add_offset() with add_tail=True. Denoted
        as self << offset.

        Parameters
        ----------
        offset : float
            Offset value with the same unit as x.

        Returns
        -------
        Wave
            Offsetted result with a new reference.

        """
        return self.add_offset(offset, add_tail=True)

    # auxiliary methods
    def iscomposite(self):
        """
        Check if the wave is directly generated by a shaping function.

        Returns
        -------
        boolean
            True if a wave is not directly generated by a shaping function.

        """
        # check if composite
        return any([isinstance(
            self.variables[key], Wave) for key in self.variables.keys()])

    def reverse(self):
        """
        Reverse y array of a wave.

        Returns
        -------
        Wave
             Modified wave object with a new reference.

        """
        properties = {'name': self.properties['name'],
                      'variables': self.properties['variables'],
                      'y': self.properties['y'][::-1],
                      'x': self.properties['x']
                      }
        return Wave(properties=properties)


    def differentiate(self):
        return


    def toWaveform(self, appendOverlap=True):
        """
        Shorthand conversion to waveform object.

        Parameters
        ----------
        appendOverlap : boolean, optional
            Determine how the sucessive waveform manipulation guideline, True
            for the waveform is appended with overlapping mode. The default is
            True.

        Returns
        -------
        Waveform
            New waveform object.

        """
        return Waveform([self], '', appendOverlap)


class Waveform(GenericWave):

    def __init__(self, waveObjList=[], name='', appendOverlap=True):
        """
        A interface to manipulate the order of waves.

        Parameters
        ----------
        waveObjList : list, optional
            Ordered list of wave object to be compiled into waveform (
            pulse train). The default is [].
        name : string, optional
            Name of waveform. The default is ''.
        appendOverlap : boolean, optional
            Set if the waveform is appended with overlapping mode or not, which
            merges the head and tail of two waves by taking an average (sum and
            divide by 2) betweenthem. If not the 2 waves are directly appended.
            The default is True.

        Returns
        -------
        Waveform
            New waveform object.

        """
        self.__waveList = list(waveObjList)
        self.name = name
        self.__appendOverlap = appendOverlap
        self.y, self.x = Waveform._synthesize(
            self.__waveList, appendOverlap=self.__appendOverlap)
        self.dx = self.x[1] - self.x[0]

    # set and get methods
    def set_waveList(self, waveObjList):
        """
        Set method for self.__waveList, which contains the element waves of the
        waveform.

        Parameters
        ----------
        waveObjList : list
            A list of wave objects to be used to setup the waveform.

        Returns
        -------
        None.

        """
        if not waveObjList:
            print('Error: waveObjList cannot be empty')
            return
        self.__waveList = waveObjList
        self.y, self.x = Waveform._synthesize(
            self.get_waveList(), appendOverlap=self.get_appendOverlap())

    def get_waveList(self):
        """
        Get method for self.__waveList, which contains the element waves of the
        waveform.

        Returns
        -------
        list
            A list of wave objects.

        """
        return self.__waveList

    def set_appendOverlap(self, toggle=True):
        """
        Set method for self.__appendOverlap.

        Parameters
        ----------
        toggle : boolean, optional
            True to enable overlapping mode. The default is True.

        Returns
        -------
        None.

        """
        self.__appendOverlap = toggle
        self.set_waveList(self.get_waveList())

    def get_appendOverlap(self, show):
        """
        Get method for self.__appendOverlap.

        Returns
        -------
        boolean
            True if the overlapping mode is enabled.

        """
        if show:
            print(self.__appendOverlap)
        return self.__appendOverlap

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
               f"overlapping mode: {self.get_appendOverlap()}\n" + \
               f"dx: {self.dx}\n" + \
               f"ID: {id(self)}"

    # algeberic methods
    def __invert__(self):
        """
        Shorthand conversion for toQubitChannel() method. Denoted as ~self.

        Returns
        -------
        QubitChannel
             New QubitChannel object with a new reference.

        """
        return self.toQubitChannel()

    def __add__(self, waveObjList):
        """
        Addition of 2 waveform objects. Denoted as self + waveObjList.

        Parameters
        ----------
        waveObjList : list or Waveform
            List of wave objects or Waveform.

        Returns
        -------
        Waveform
            Appended waveform with a new reference.

        """
        waveObjList = Waveform._toWaveObjList(waveObjList)
        waveList_new = self.get_waveList() + waveObjList
        return Waveform(waveList_new, appendOverlap=self.get_appendOverlap())

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
            Appended waveform with a new reference.

        """
        waveList_new = self.get_waveList() * num
        return Waveform(waveList_new, appendOverlap=self.get_appendOverlap())

    def __rmul__(self, other):
        # Image method for __mul__
        # https://www.cnblogs.com/scolia/p/5686267.html
        return self * other

    def __rshift__(self, waveform):
        """
        Shorthand operator for self.alignwith() with use_1st_head=True and
        align_2nd_head=False. Denoted as self >> waveform.

        Parameters
        ----------
        waveform : Waveform
            Waveform to be aligned.

        Returns
        -------
        Waveform
            Aligned result of Waveform with the same reference.

        """
        self.alignwith(waveform, use_1st_head=True, align_2nd_head=False)
        return self

    def __irshift__(self, waveform):
        """
        Shorthand operator for self.alignwith() with use_1st_head=False and
        align_2nd_head=False. Denoted as self >>= waveform.

        Parameters
        ----------
        waveform : Waveform
            Waveform to be aligned.

        Returns
        -------
        Waveform
            Aligned result of Waveform with the same reference.

        """
        self.alignwith(waveform, use_1st_head=False, align_2nd_head=False)
        return self

    def __lshift__(self, waveform):
        """
        Shorthand operator for self.alignwith() with use_1st_head=False and
        align_2nd_head=True. Denoted as self << waveform.

        Parameters
        ----------
        waveform : Waveform
            Waveform to be aligned.

        Returns
        -------
        Waveform
            Aligned result of Waveform with the same reference.

        """
        self.alignwith(waveform, use_1st_head=False, align_2nd_head=True)
        return self

    def __ilshift__(self, waveform):
        """
        Shorthand operator for self.alignwith() with use_1st_head=True and
        align_2nd_head=True. Denoted as self <<= waveform.

        Parameters
        ----------
        waveform : Waveform
            Waveform to be aligned.

        Returns
        -------
        Waveform
            Aligned result of Waveform with the same reference.

        """
        self.alignwith(waveform, use_1st_head=True, align_2nd_head=True)
        return self

    # math methods
    def permute(self, order):
        """
        Re-ordering the wave object list of a waveform.

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
        Insert a list of wave objects to a waveform with specified indices.

        Parameters
        ----------
        indices : list, optional
            Indices to be inserted with wave. The default is [].
        waveObjList : list, optional
            List of wave objects to be inserted. The default is [].

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
        Replace a list of wave objects to a waveform with specified indices.

        Parameters
        ----------
        indices : list, optional
            Indices to be replaced with wave. The default is [].
        waveObjList : list, optional
            List of wave objects to be replaced. The default is [].

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
        len_longer = longer.get_span()
        len_shorter = shorter.get_span()
        appendLength = abs(len_longer - len_shorter) if not (
            use_1st_head ^ align_2nd_head) else len_longer
        # modify each length
        if use_1st_head ^ align_2nd_head:
            longer._offset(len_shorter, shorter is self if use_1st_head
                           else longer is self)
            shorter._offset(appendLength, longer is self if use_1st_head
                            else shorter is self)
        else:
            shorter._offset(appendLength, use_1st_head)

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
        return [waveObj.toWaveform(
                appendOverlap) for waveObj in self.get_waveList()]

    def toQubitChannel(self):
        """
        Shorthand conversion to QubitChannel object.

        Returns
        -------
        QubitChannel
            New QubitChannel object.

        """
        return QubitChannel(self)

    def _offset(self, offset=0., add_tail=False, reset=False):
        """
        Backend function to offset the first or the last wave object in a
        waveform.

        Parameters
        ----------
        offset : float, optional
            Offset by a value with the same unit as x. The default is 0.
        add_tail : boolean, optional
            True to append 0s at tail, at head otherwise. The default is False.
        reset : boolean, optional
            Remove the offset function. The default is False.

        Returns
        -------
        None.

        """
        waveObj = self.get_waveList()[-1 if add_tail else 0]
        waveObj_new = waveObj.add_offset(offset, add_tail, reset)
        self.replace([-1 if add_tail else 0], [waveObj_new])

    @classmethod
    def _synthesize(cls, waveList, appendOverlap=False):
        """
        Backend function to compile the list of wave objects into a complete
        waveform (pulse train).

        Parameters
        ----------
        cls : Waveform class
            Waveform class object.
        waveList : list
            List of wave objects to compiled into waveform.
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
                y = waveObj.y
                x = waveObj.x
                offset = x[-1]
                try:
                    dx = x[1] - x[0]
                except(IndexError):
                    dx = 0
                continue
            if appendOverlap:
                y = np.hstack([
                    y[:-1], np.array([(y[-1] + waveObj.y[0])/2]), waveObj.y[1:]
                    ])
                x = np.hstack([x, waveObj.x[1:] + offset])
            else:
                x = np.hstack([x, waveObj.x + offset + dx])
                y = np.hstack([y, waveObj.y])
            offset = x[-1]
        return y, x

    @classmethod
    def _toWaveObjList(cls, waveform):
        """
        Backend function to transform the input into a list of wave objects.

        Parameters
        ----------
        cls : Waveform class
            Waveform class object.
        waveform : Waveform or Wave
            Waveform or Wave object.

        Returns
        -------
        list
            List of wave objects.

        """
        if isinstance(waveform, Waveform):
            return waveform.get_waveList()
        if isinstance(waveform, Wave):
            return [waveform]
        else:
            return waveform


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
        self.wire_names = [waveform.get_name() for waveform in self.wires]
        QubitChannel._align(self)
        self.x = self.wires[0].get_x()
        self.dx = self.x[1] - self.x[0]
        self.y = [self.wires[i].get_y() for i in range(self.number_of_wires)]
        self.name = ''

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
        fig.suptitle(self.get_name(), fontsize=16)
        for index, i in zip(wire_indices, range(num_plot)):
            axisObjArr += [plt.subplot(num_plot, 1, i+1)]
            self.wires[i].plot()
            nameList += [self.wire_names[index]]
            axisObjArr[i].legend([nameList[i]], loc="best")
        print("plot size=["+str(size[0])+","+str(size[1])+"]")
        return axisObjArr

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

    def get_wire_names(self, show):
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
