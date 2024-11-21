import os

if '__main__' == __name__:
    keys = sorted(os.environ.keys())
    for k in keys:
        print(f"|{k}|", f"|{os.environ[k]}|")
