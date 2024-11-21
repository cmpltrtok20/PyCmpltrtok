import datetime
import pickle
import sys
import random
import re
import io
from contextlib import redirect_stdout
import json

STATES_FILE_PATH = 'choose_student.tmp.pkl'


def clone(xobj):
    """
    克隆对象
    https://stackoverflow.com/questions/22281059/set-object-is-not-json-serializable

    :param xobj: 原对象
    :return: 克隆后的对象
    """
    if isinstance(xobj, set):
        return set(list(xobj))
    xclone = json.loads(json.dumps(xobj))
    return xclone


def separator(*args):
    """画个分割线，在中间打印所有参数"""
    if not len(args):
        args = ('', )
    print('-------------------------------%s----------------------------------' % (*args, ))


def load_names(xfile_name, xname, silent=False):
    """
    读取名单

    :param xfile_name: 文件名
    :param xname: 本文件的意义
    :param silent: 是否不打印信息

    :return
    """
    if not silent:
        separator(xname)

    with open(xfile_name, encoding='gbk') as f:
        xnames = f.readlines()
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
        print('候选者名单中有重复的名字！为保证每个名字的概率等同，每个名字只读取1次。如果确实有重名的情况，请通过加后缀等方式区别开。重名情况如下：')
        xname2cnt = {}
        for xname in xnames_raw:
            xcnt = xname2cnt.get(xname, 0)
            xname2cnt[xname] = xcnt + 1
        for xname, xcnt in xname2cnt.items():
            if xcnt > 1:
                print(xname, 'x', xcnt)

    if not silent:
        print(xnames)
        print('共%d人' % (n_all, ))
        separator()
    return xnames


if '__main__' == __name__:

    def _main():

        def load_all_and_candidates(silent=False):
            """
            读取”所有同学“和”本次候选名单“。

            :return: 失败时返回None, 成功时返回tuple(本次候选list, 本次候选set, 所有同学list, 所有同学set, 所有同学数目)
            """
            xnames_all = load_names('choose_student.all.txt', '所有同学', silent)
            xnames_all_set = set(xnames_all)
            n_all = len(xnames_all)

            xnames = load_names('choose_student.tmp.txt', '本次候选名单', silent)
            xnames_set = set(xnames)

            # 错误的情况
            xnew = sorted(xnames_set - xnames_all_set)
            if xnew:
                print(
                    '以下名字在本次候选名单中，但不存在于“所有同学”的列表中！请移除错误的候选者、添加新名字、或进行其他操作来纠正错误，然后重新运行程序')
                print(xnew)
                return None

            return xnames, xnames_set, xnames_all, xnames_all_set, n_all

        # 进入后读取”所有同学“和”本次候选名单“，直到成功
        while True:
            xresult = load_all_and_candidates()
            if xresult is None:
                print('调整名单后，按回车键继续……')
                input()
                continue
            xnames, xnames_set, xnames_all, xnames_all_set, n_all = xresult
            break

        # 读入或新建历史状态
        try:
            with open(STATES_FILE_PATH, 'br') as f:
                xstates = pickle.load(f)
        except Exception as ex:
            # print(ex, file=sys.stderr)
            xstates = {
                'history': [[], ],
                'n_chosen': 0,
                'n_all': n_all,
            }

        def save_states():
            n_chosen = len(xstates['history'][-1])
            xstates['n_chosen'] = n_chosen
            xstates['n_all'] = n_all
            with open(STATES_FILE_PATH, 'bw') as f:
                pickle.dump(xstates, f)

        def input_cmd():
            """
            打印提示，获取用户键盘输入。

            :return: 用户键入内容（去除两端空白和回车，全转小写。）
            """
            xprompt = """\
>>>> (%d/%d)
c - 选择同学
p - 打印本轮选择
h - 打印所有历史概况
"s 历史序号" - 打印指定历史（例如"s 0", "s 1", "s 2", "s -1", "s -2", ...） （请不要加引号）
L - 修改列表后，重新加载
q - 退出: """ % (len(xstates['history'][-1]), n_all, )
            print(xprompt)
            xinput = input().strip().lower()
            if not xinput:
                return None
            return xinput

        def print_history(xindex):
            try:
                xhistories = [xhistory for xhistory in xstates['history'] if xhistory]
                xhistory = xhistories[xindex]
            except IndexError as ex:
                print('索引%d超出范围！' % (xindex, ))
                return
            for xchosen, xdt in xhistory:
                print(xdt, '>>>>', xchosen)

        xregexp_s_d = re.compile(r's\s+(-?\d+)', re.I)
        xoutput = ''
        xfirst = True

        while True:
            # 保存状态
            save_states()

            # 读取”所有同学“和”本次候选名单“，直到成功
            if not xfirst:
                xnames, xnames_set, xnames_all, xnames_all_set, n_all = load_all_and_candidates()  # 因为程序进入已经加载了并打印了，在此第一次加载不打印
            else:
                xfirst = False
            if xresult is None:
                print('调整名单后，按回车键继续……')
                input()
                continue

            # 保存状态 （因n_all可能因上面的读取而改变，因此再保存一下。）
            save_states()

            # 如果本轮选择已经覆盖所有同学，则开辟下一轮
            xhistory_names_set = set([xname for xname, xdt in xstates['history'][-1]])
            xleft_names_set = xnames_all_set - xhistory_names_set
            xleft_names = sorted(xleft_names_set)
            n_left = len(xleft_names_set)

            # 本轮结束，下一轮开始！
            if not xleft_names_set:
                separator('本轮结束，下一轮开始！')
                xstates['history'].append([])
                xleft_names_set = clone(xnames_all_set)
                xleft_names = sorted(xleft_names_set)
                n_left = len(xleft_names_set)
                xavailable_set = clone(xleft_names_set)
                xavailable = sorted(xavailable_set)
                n_availabe = len(xavailable_set)
            else:
                # 有效的候选者（候选者中本轮没有选择过的同学）
                xavailable_set = xleft_names_set.intersection(xnames_set)
                xavailable = sorted(xavailable_set)
                n_availabe = len(xavailable_set)

            # 输出候选同学
            separator('候选同学')
            print(xleft_names)
            print('共%d人' % (n_left,))

            # 输出有效的候选者
            separator('有效的候选者')
            if not xavailable:
                print('候选者里没有可以选的人了，请添加候选者，然后输入L进行重新加载！')
            else:
                print(xavailable)
                print('共%d人' % (n_availabe,))

            # 输出缓存
            print(xoutput)

            # 读取用户输入
            xinput = input_cmd()

            # 处理用户命令，并缓存输出到变量
            # https://stackoverflow.com/questions/1218933/can-i-redirect-the-stdout-into-some-sort-of-string-buffer
            with io.StringIO() as buf, redirect_stdout(buf):

                def collect_stdout():
                    """收集stdout缓存"""
                    nonlocal xoutput
                    xoutput = buf.getvalue()  # 缓存

                # 选择同学
                if 'c' == xinput:
                    if not n_availabe:
                        print('请增加候选者，输入L进行重新加载。')
                        collect_stdout()
                        continue
                    xindex = random.randint(0, n_availabe - 1)
                    xchosen = xavailable[xindex]
                    xdt = datetime.datetime.now()
                    print('>>>>', xdt, xchosen)

                    xstates['history'][-1].append((xchosen, xdt))

                # 打印本轮选择
                elif 'p' == xinput:
                    if not xstates['history'][-1]:
                        print('本轮选择还没有进行。')
                        collect_stdout()
                        continue
                    print('本轮选择历史：')
                    print_history(-1)

                # 打印所有历史概况
                elif 'h' == xinput:
                    for i, xone in enumerate(xstates['history']):
                        if not xone:
                            continue
                        print('Index =', i, '(共', len(xone), '位) from', xone[0][1], 'to', xone[-1][1])

                # 重新加载列表
                elif 'l' == xinput:
                    collect_stdout()
                    continue  # 开始新的一轮，自动就重新加载了

                # 退出
                elif 'q' == xinput:
                    collect_stdout()
                    print(xoutput)
                    break

                # 打印指定历史
                else:
                    xmatch = xregexp_s_d.search(xinput)
                    if xmatch is None:
                        print('输入命令错误！')
                        collect_stdout()
                        continue
                    xindex = int(xmatch[1])
                    print_history(xindex)

                collect_stdout()

    _main()
