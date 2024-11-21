import copy

if '__main__' == __name__:

    def _main():
        with open('choose_student.all.txt', 'r', encoding='gbk') as fi:
            xnames = fi.readlines()
            # print(xnames)

            # 去除空白和回车
            xnames = [xname.strip() for xname in xnames]
            # print(xnames)

            # 去除空行和注释
            xnames_raw = [xname for xname in xnames if xname and xname[0] != '#']
            # print(xnames_raw)

            # 去重排序
            xnames = sorted(set(xnames_raw))
            n_all_raw = len(xnames_raw)
            n_all = len(xnames)

            # 提示重名的情况
            if n_all_raw != n_all:
                print(
                    '候选者名单中有重复的名字！为保证每个名字的概率等同，每个名字只读取1次。如果确实有重名的情况，请通过加后缀等方式区别开。重名情况如下：')
                xname2cnt = {}
                for xname in xnames_raw:
                    xcnt = xname2cnt.get(xname, 0)
                    xname2cnt[xname] = xcnt + 1
                for xname, xcnt in xname2cnt.items():
                    if xcnt > 1:
                        print(xname, 'x', xcnt)

            xnames_all = copy.deepcopy(xnames)
            xnames_all_set = set(xnames_all)

        with open('choose_student.set.tmp.txt', 'r', encoding='gbk') as fi:
            xnames = fi.readlines()
            # print(xnames)

            # 去除空白和回车
            xnames = [xname.strip() for xname in xnames]
            # print(xnames)

            # 去除空行和注释
            xnames_raw = [xname for xname in xnames if xname and xname[0] != '#']
            # print(xnames_raw)

            # 去重排序
            xnames = sorted(set(xnames_raw))
            n_all_raw = len(xnames_raw)
            n_all = len(xnames)

            # 提示重名的情况
            if n_all_raw != n_all:
                print(
                    '候选者名单中有重复的名字！为保证每个名字的概率等同，每个名字只读取1次。如果确实有重名的情况，请通过加后缀等方式区别开。重名情况如下：')
                xname2cnt = {}
                for xname in xnames_raw:
                    xcnt = xname2cnt.get(xname, 0)
                    xname2cnt[xname] = xcnt + 1
                for xname, xcnt in xname2cnt.items():
                    if xcnt > 1:
                        print(xname, 'x', xcnt)

            xnames_set = set(xnames)

        xrest = xnames_all_set - xnames_set
        print('REST:', sorted(xrest))
        xwrong = xnames_set - xnames_all_set
        print('WRONG:', sorted(xwrong))

    _main()
