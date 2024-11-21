from PyCmpltrtok.common import translate_s
import sys

if '__main__' == __name__:
    s = 1676445890
    if len(sys.argv) >= 2:
        s = int(sys.argv[1])
    xstr = translate_s(s)
    print(s, xstr)
