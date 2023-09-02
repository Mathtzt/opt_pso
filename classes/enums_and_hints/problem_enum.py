from enum import Enum

class ProblemFuncNames(Enum):
    F1_BASIC = 'f1_basic'
    F1 = 'f1'
    F2 = 'f2'
    F4 = 'f4'
    F6 = 'f6'
    F7 = 'f7'
    F8_BASIC = 'f8_basic'
    F8 = 'f8'
    F9 = 'f9'
    F14 = 'f14'

class ProblemFuncOptValue(Enum):
    DEFAULT = 0. 
    F1_BASIC = 0.
    F1 = 100.
    F2 = 200.
    F4 = 400.
    F6 = 600.
    F7 = 700.
    F8_BASIC = 0.
    F8 = 800.
    F9 = 900.
    F14 = 1400.