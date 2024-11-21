"""
https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
WWW: how to iterate a pandas series by columns?
"""

import pandas as pd
from PyCmpltrtok.common import sep

if '__main__' == __name__:

    def _main():
        xstd_path = r'D:\_dell7590_root\sync\0_bdn\note\note_root\aa\teach in baway\process\2103A-NLP\专高6_2103A\考试\2103A-NLP N5 calc-std.xlsx'
        xtech_path = r'D:\_dell7590_root\sync\0_bdn\note\note_root\aa\teach in baway\process\2103A-NLP\专高6_2103A\考试\2103A-NLP N5 calc-tech.xlsx'
        xtheory_path = r'D:\_dell7590_root\sync\0_bdn\note\note_root\aa\teach in baway\process\2103A-NLP\专高6_2103A\考试\2103A-NLP N5 calc-theory.xlsx'

        sep('Forced up')
        xset_forced = set([
            '魏绍伦', '王钰', '贺丞惺', '董俊杰', '吴佳佳', '杨海军', '丛昊毓',
        ])

        sep('std')
        xstd = pd.read_excel(xstd_path, header=None)
        xstd = xstd.reset_index()  # make sure indexes pair with number of rows
        # print(xstd)
        xmap2thresh_tech, xmap2thresh_theory = {}, {}
        xmap2is_dbl70 = {}
        for index, xrow in xstd.iterrows():
            xname = xrow[0]
            xmin_theory = min(float(xrow[5]), float(xrow[7]))
            xmin_tech = min(float(xrow[6]), float(xrow[8]))
            x3 = int(xrow[3])
            print(xname, xmin_theory, xmin_tech, x3)
            xmap2thresh_tech[xname] = xmin_tech
            xmap2thresh_theory[xname] = xmin_theory
            xmap2is_dbl70[xname] = x3 == 140
        # print(xmap2is_dbl70)

        sep('tech')
        xtech = pd.read_excel(xtech_path, header=None)
        xtech = xtech.reset_index()
        print(xtech)
        idx2name, idx2score = {}, {}
        for index, xrow in xtech.iterrows():
            if 0 == index:
                for col, item in xrow.items():
                    if 'index' == col:
                        continue
                    # print(col, item)
                    idx2name[col] = item
            elif 1 == index:
                for col, item in xrow.items():
                    if 'index' == col:
                        continue
                    # print(col, item)
                    idx2score[col] = float(item)
        xmap2tech = {}
        for idx, name in idx2name.items():
            xmap2tech[name] = idx2score[idx]
        print(xmap2tech)

        sep('theory')
        xtheory = pd.read_excel(xtheory_path, header=None)
        xtheory = xtheory.reset_index()
        # print(xtheory)
        xmap2theory = {}
        for index, xrow in xtheory.iterrows():
            xmap2theory[xrow[0]] = float(xrow[7])
        print(xmap2theory)

        passed, unpassed = [], []
        dbl70 = []
        for xname, xmin_tech in xmap2thresh_tech.items():
            xmin_theory = xmap2thresh_theory[xname]
            xtech = xmap2tech[xname]
            xtheory = xmap2theory[xname]
            if xtech >= xmin_tech and xtheory >= xmin_theory:
                passed.append(xname)
            elif xtech >= 70 and xtheory >=70 and xmap2is_dbl70[xname]:
                dbl70.append(xname)
            else:
                unpassed.append(xname)
        passed = sorted(passed)
        unpassed = sorted(unpassed)
        sep('passed')
        print(len(passed), passed)
        sep('unpassed')
        print(len(unpassed), unpassed)
        sep('double 70')
        print(len(dbl70), dbl70)
        sep('unpassed detail')
        for xname in unpassed:
            print(xname, xmap2theory[xname], xmap2tech[xname],
                  '(', xmap2thresh_theory[xname], xmap2thresh_tech[xname], ')',
                  '*' if xname in xset_forced else '')
        sep('double 70 detail')
        for xname in dbl70:
            print(xname, xmap2theory[xname], xmap2tech[xname])

    _main()
