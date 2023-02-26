import numpy as np
from scipy.linalg.blas import dgemm
from scipy.linalg.lapack import dgetrf, dgetrs

def lsame(ca, cb):
    # Test if the characters are equal
    if ca == cb:
        return True

    # Now test for equivalence if both characters are alphabetic.
    zcode = ord('Z')

    # Use 'Z' rather than 'A' so that ASCII can be detected on Prime
    # machines, on which ord returns a value with bit 8 set.
    # ord('A') on Prime machines returns 193 which is the same as
    # ord('A') on an EBCDIC machine.
    inta = ord(ca)
    intb = ord(cb)

    if zcode == 90 or zcode == 122:
        # ASCII is assumed - ZCODE is the ASCII code of either lower or
        # upper case 'Z'.
        if 97 <= inta <= 122:
            inta -= 32
        if 97 <= intb <= 122:
            intb -= 32
    elif zcode == 233 or zcode == 169:
        # EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or
        # upper case 'Z'.
        if 129 <= inta <= 137 or 145 <= inta <= 153 or 162 <= inta <= 169:
            inta += 64
        if 129 <= intb <= 137 or 145 <= intb <= 153 or 162 <= intb <= 169:
            intb += 64
    elif zcode == 218 or zcode == 250:
        # ASCII is assumed, on Prime machines - ZCODE is the ASCII code
        # plus 128 of either lower or upper case 'Z'.
        if 225 <= inta <= 250:
            inta -= 32
        if 225 <= intb <= 250:
            intb -= 32

    return inta == intb

