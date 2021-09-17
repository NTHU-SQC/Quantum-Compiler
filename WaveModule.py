# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 20:18:43 2021

for set / get vs IDE:
    https://stackoverflow.com/questions/52312897

@author: Alaster
"""

import TemplateModule as tpm
import numpy as np
from copy import deepcopy
from ShapeModule import parse, setFunc


class Wave(tpm.GenericWave):

    def __init__(self, generator=None, properties={}):
        """
        Encapsulation of outputs generated by user defined function (shape
        function) with the capability to further modify wave's amplitude, shape
        and phase.

        Parameters
        ----------
        generator : dict, optional
            Discription dict for wave generation. Use None to enable explicit
            content creation of Wave object. The default is None.
        properties : dict, optional
            Use explicit assignment to create Wave object, available only if
            generator is None. The default is {}.

        Returns
        -------
        Wave
            An encapsulated Wave object.

        """
        if generator is None:
            temp = deepcopy(properties)
            self._x = temp['x']
            self._y = temp['y']
            self._name = temp['name']
            self._appendRule = temp['appendRule']
            return
        self._x, self._y, self._name, self._appendRule = parse(generator)
        

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
                        both the last 2nd point (fisrt wave) and the first
                        point (second wave) are kept with an addition offset dx
                        is set between them. In this mode the last point of the
                        first wave is always neglected to preserve total span
                        consistency.

        """
        self._appendRule = appendRule

    def __invert__(self):
        """
        Shorthand conversion to waveform object. Denoted as ~self.

        Returns
        -------
        Waveform
            New waveform object.

        """
        return Waveform([self], self.name)

    def __neg__(self):
        """
        Reverse y array of a wave. Denoted as -self.

        Returns
        -------
        Wave
             Object with a new reference.

        """
        properties = {'name': self.name,
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
                          'y': self.y + waveObj,
                          'x': self.x,
                          'appendRule': self.appendRule
                          }
            return Wave(properties=properties)
        longer = max(self, waveObj)
        if self is waveObj:
            shorter = self
        else:
            shorter = [obj for obj in [self, waveObj] if obj is not longer][0]
        x, length = longer.x, len(shorter)
        y = np.concatenate((
            longer.y[:length] + shorter.y[:length], longer.y[length:]
            ))
        properties = {'name': self.name,
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
                          'y': self.y - waveObj,
                          'x': self.x,
                          'appendRule': self.appendRule
                          }
            return Wave(properties=properties)
        longer = max(self, waveObj)
        if self is waveObj:
            shorter = self
        else:
            shorter = [obj for obj in [self, waveObj] if obj is not longer][0]
        x, length = longer.x, len(shorter)
        y = np.concatenate((
            longer.y[:length] - shorter.y[:length], longer.y[length:]
            ))
        properties = {'name': self.name,
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
                          'y': self.y * waveObj,
                          'x': self.x,
                          'appendRule': self.appendRule
                          }
            return Wave(properties=properties)
        longer = max(self, waveObj)
        if self is waveObj:
            shorter = self
        else:
            shorter = [obj for obj in [self, waveObj] if obj is not longer][0]
        x, length = longer.x, len(shorter)
        y = np.concatenate((
            longer.y[:length] * shorter.y[:length], longer.y[length:]
            ))
        properties = {'name': self.name,
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
            properties = {'name': self.faxisname,
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

    def __str__(self):
        """
        Print the status of a wave. Denoted as print(self).

        Returns
        -------
        string
            The status of a wave.

        """
        return f"name: {self.name}\n" + \
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
                      'y': abs(self.y),
                      'x': self.x,
                      'appendRule': self.appendRule
                      }
        return Wave(properties=properties)


class Waveform(tpm.GenericWave):

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
        Shorthand conversion to QubitChannel object. Denoted as ~self.

        Returns
        -------
        QubitChannel
            New QubitChannel object.

        """
        return QubitChannel(self)

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
        Shorthand operator for self.offset() with positive offset. Denoted
        as self >> offset.

        Parameters
        ----------
        offset : float
            Offset value with the same unit as x.

        Returns
        -------
        Waveform
            Offsetted result with a new reference.

        """
        return self.offset(offset)

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
        Shorthand operator for self.offset() with negative offset. Denoted as
        self << offset.

        Parameters
        ----------
        offset : float
            Offset value with the same unit as x.

        Returns
        -------
        Waveform
            Offsetted result with a new reference.

        """
        return self.offset(-offset)

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
        samp_rate = self.df
        if use_1st_head ^ align_2nd_head:
            addListA = self.__class__._nullBlock(
                self.span, samp_rate, self.appendRule)
            addListB = self.__class__._nullBlock(
                waveform.span, samp_rate, waveform.appendRule)
            if use_1st_head:
                # print('T-F')
                self.waveList = addListB + self.waveList
                waveform.waveList = waveform.waveList + addListA
            else:
                # print('F-T')
                self.waveList = self.waveList + addListB
                waveform.waveList = addListA + waveform.waveList
        else:
            span = round(abs(self.span - waveform.span),
                         self.__class__.EFF_TIME_DIGIT)
            # distinguish longer & shorter waveform
            longer = max(self, waveform)
            shorter = min(self, waveform)
            if use_1st_head:
                # print('T-T')
                addList = self.__class__._nullBlock(
                    span, samp_rate, [True, longer.appendRule[-1]])
                shorter.waveList = shorter.waveList + addList
            else:
                # print('F-F')
                addList = self.__class__._nullBlock(
                    span, samp_rate, [longer.appendRule[0], True])
                shorter.waveList = addList + shorter.waveList

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
        offset = round(offset, self.__class__.EFF_TIME_DIGIT)
        if abs(offset) < self.dx:
            return self
        span = abs(offset)
        samp_rate = self.df
        if offset > 0:
            return Waveform(
                self.__class__._nullBlock(
                    span, samp_rate, [self.appendRule[0], False]
                    ) + self.waveList
                )
        else:
            return Waveform(
                self.waveList + self.__class__._nullBlock(
                    span, samp_rate, [False, self.appendRule[-1]]
                    )
                )

    def fill_total_point(self, total_point=0):
        add_point = total_point - len(self)
        if add_point <= 0:
            return self
        span = round((add_point + 1) * self.dx, self.__class__.EFF_TIME_DIGIT)
        return self << span

    @classmethod
    def _nullBlock(cls,
                   span=.0,
                   sampling_rate=1e9,
                   appendRule=[False, False]):
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
        generator = setFunc(
            'const', [0], span, sampling_rate, 'null', appendRule
            )
        return [Wave(generator)]

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
        delEnd = 0
        for waveObj in waveList:
            if waveObj.x.size == 0:  # skip null waves
                continue
            if x.size == 0:  # initial null filling
                y = waveObj.y
                x = waveObj.x
                offset = x[-1]
                previous = waveObj
                continue
            # concatenate according to appendrules
            leftRule = previous.appendRule[1]
            rightRule = waveObj.appendRule[0]
            if leftRule and rightRule:
                delEnd = 1
            if leftRule ^ rightRule:
                if leftRule:
                    # print('T-F')
                    y = np.hstack([y[:len(y)-delEnd], waveObj.y[1:]])
                else:
                    # print('F-T')
                    y = np.hstack([y[:-1-delEnd], waveObj.y])
                x = np.hstack([x[:len(x)-delEnd], waveObj.x[1:] + offset])
            else:
                if leftRule:
                    # print('T-T')
                    y = np.hstack([y[:len(y)-delEnd], waveObj.y])
                    x = np.hstack(
                        [x[:len(x)-delEnd], waveObj.x + offset]
                        )
                else:
                    # print('F-F')
                    y = np.hstack([y[:-1-delEnd],
                                   np.array([(y[-1-delEnd] + waveObj.y[0])/2]),
                                   waveObj.y[1:]])
                    x = np.hstack([x[:len(x)-delEnd], waveObj.x[1:] + offset])
            offset = x[-1]
            previous = waveObj
            delEnd = 0
        return y[:len(y)-delEnd], np.round(
            x[:len(x)-delEnd], cls.EFF_TIME_DIGIT
            )

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


class QubitChannel(tpm.GenericWave):

    def __init__(self, *waveforms):
        """
        An interface for single qubit waveform control. Each waveform is
        assigned to an independent wire and alignment is performed among wires.

        Parameters
        ----------
        *waveforms : Waveform
            Waveforms to be assigned to wires.

        Returns
        -------
        QubitChannel
            New QubitChannel object.

        """
        self._wires = np.array(waveforms)
        self._wire_names = [waveform.name for waveform in self._wires]
        self.__class__.align(self)
        self._name = ''

    @property
    def x(self):
        self._x = self._wires[0].x
        return self._x

    @property
    def y(self):
        self._y = [waveform.y for waveform in self._wires]
        return self._y

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
                '' for i in range(len(self._wire_names) - len(nameList))
                ]

    def get_wire(self, wire_name=''):
        """
        Get Waveform object from specified wirename.

        Parameters
        ----------
        wire_name : string, optional
            Name of the wire. The default is ''.

        Returns
        -------
        Waveform
            Corresponding Waveform object.

        """
        idx = self._wire_names.index(wire_name)
        return self._wires[idx]

    def __str__(self):
        """
        Print the status of a QubitChannel object.

        Returns
        -------
        string
            The status of a QubitChannel object.

        """
        return f"name: {self.name}\n" + \
            f"wire names: {self._wire_names}\n" + \
            f"point number: {len(self)}\n" + \
            f"span: {self.span}\n" + \
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
            raise ValueError('Wire concatenation with unequal size arrays')
        temp = QubitChannel(*(self._wires + qcObj._wires))
        temp.wire_names = self.wire_names
        return temp

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
        temp = QubitChannel(*(self._wires * num))
        temp.wire_names = self.wire_names
        return temp

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
        waveformList : list or QubitChannel or Waveform
            List of Waveforms (or a Waveform object) or a QubitChannel object.

        Returns
        -------
        QubitChannel
            Appended QubitChannel object with a new reference.

        """
        waveformList = self.__class__._toWaveformObjList(waveformList)
        return QubitChannel(*self._wires, *waveformList)

    def add_null_wire(self, *wireIndex):
        """
        Add null wires to QubitChannel object.

        Parameters
        ----------
        *wireIndex : int
            Indices to be inserted with null waveform. Must smaller than the
            number of wires.

        Returns
        -------
        QubitChannel
            Appended QubitChannel object with a new reference.

        """
        nullblock = Waveform._nullBlock(self.span, self.df)
        wires = np.insert(self._wires, wireIndex, nullblock)
        return QubitChannel(*wires)

    def plot(self,
             wire_indices=[],
             size=[6.4, 4.8],
             figure_name='',
             allInOne=False,
             toByteStream=False,
             showSizeInfo=True
             ):
        """
        A quick plot among wires of a QubitChannel object.

        Parameters
        ----------
        wire_indices : list, optional
            List of indices of wires to be examined. The default is [].
        size : list, optional
            Size of each subplot. The default is [6.4, 4.8].
        figure_name : str, optional
            Name of figure. The default is ''.
        allInOne : bool, optional
            Set True to put all traces into the same subplot. The default is
            False.
        toByteStream : bool, optional
            Set True to convert plot into byte stream without plotting. The
            default is False.
        showSizeInfo : bool, optional
            Set True to show plot size during plot creation. The default is
            True.

        Returns
        -------
        fig/BytesIO : matplotlib.lines.Line2D or BytesIO object
            Figure of byte stream object.

        """
        if not wire_indices:
            wire_indices = range(len(self._wires))
        ydict_list = [{}] * len(wire_indices)
        for idx, i in zip(wire_indices, range(len(wire_indices))):
            ydict_list[i] = tpm.axis(
                self._wire_names[idx], 'amplitude', self._wires[idx].y, False
                )
        if not figure_name:
            figure_name = self.name
        return tpm.draw(
            self.xaxis, ydict_list, figure_name=figure_name, size=size,
            allInOne=allInOne, toByteStream=toByteStream,
            showSizeInfo=showSizeInfo
            )

    @classmethod
    def align(cls, qcObj, ref=None):
        """
        Backend method to perform wire-wise alignment. If no reference object
        is assigned, the longest wire in the QubitChannel object will be
        selected.

        Parameters
        ----------
        cls : QubitChannel class
            QubitChannel class.
        qcObj : QubitChannel, Waveform, Wave
            QubitChannel object with wires to be aligned.
        ref : Wave/Waveform/QubitChannel/float, optional
            Reference to be aligned with. The float datatype corresponds to
            the span while others correspond to GenericWave children class
            object. The default is None.

        Returns
        -------
        None.

        """
        if ref:
            if isinstance(ref, float):
                longest = Waveform(Waveform._nullBlock(ref, qcObj.df))
            else:
                longest = Waveform(Waveform._nullBlock(ref.span, qcObj.df))
        else:
            longest = max(qcObj._wires, key=len)
        for waveform in qcObj._wires:
            if waveform is not longest:
                longest <<= waveform

    @classmethod
    def alignQubitChannels(cls, *qcObjList):
        """
        Align all QubitChannel objects.

        Parameters
        ----------
        cls : QubitChannel
            QubitChannel class.
        *qcObjList : QubitChannel
            QubitChannel objects for alignment.

        Returns
        -------
        QubitChannel
            Aligned QubitChannel objects with original references.

        """
        if len(qcObjList) == 1:
            return *qcObjList,
        longest = max(qcObjList, key=len)
        for qcObj in qcObjList:
            cls.align(qcObj, longest)
        return *qcObjList,

    @classmethod
    def null(cls, spanRef, wireRef, default_sampling_rate=1e9):
        """
        Generate null QubitChannel object with reference objects.

        Parameters
        ----------
        cls : QubitChannel
            QubitChannel class.
        spanRef : float, QubitChannel
            Reference object to determine the span. The float datatype
            corresponds to the span while others correspond to GenericWave
            children class object.
        wireRef : int, QubitChannel
            Reference object to determine the number of wires.The int datatype
            corresponds to the number of wires while others correspond to
            GenericWave children class object.

        Returns
        -------
        QubitChannel
            QubitChannel object with a new reference.

        """
        if isinstance(spanRef, float):
            nullblock = Waveform(Waveform._nullBlock(
                spanRef, default_sampling_rate
                ))
        else:
            nullblock = Waveform(Waveform._nullBlock(
                spanRef.span, spanRef.df
                ))
        if isinstance(wireRef, int):
            wirenum = wireRef
        else:
            wirenum = len(wireRef._wires)
        return QubitChannel(*([nullblock] * wirenum))

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
    pass
