from PyCmpltrtok.common import translate_ns
import sys

if '__main__' == __name__:
    ns = 1675844638834637500
    if len(sys.argv) >= 2:
        ns = int(sys.argv[1])
    xstr = translate_ns(ns)
    print(ns, xstr)
