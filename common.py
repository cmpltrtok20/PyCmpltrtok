"""
Common routines
"""
import re
import os
import datetime
import numpy as np


def sep(label='', cnt=32, char='-', rchar=None):
    """Util function to print a separator line with label."""
    if rchar is None:
        rchar = char
    print(char * cnt, label, rchar * cnt, sep='')


_get_prefix_from_saved_path_xdict = {}  # Go along with function get_prefix_from_path


def divide_int(i):
    i = int(i)
    if i % 2 == 0:
        return i // 2, i // 2
    else:
        return i // 2, i // 2 + 1


def get_prefix_from_path(path, ext, force=True):
    """
    Example: ext=.pth, "xxx/yyy.pth" => "xxx/yyy"
    If ext=None, any ext name will be removed.
    """
    if path is None:
        return None

    if ext == '':
        return path
    elif ext is None:
        regexp = re.compile(r'(.+)\.[^\.]+')
    else:
        regexp = _get_prefix_from_saved_path_xdict.get(ext, None)
        if regexp is None:
            ext_escape = re.escape(ext)
            regexp = re.compile(r'^(.+)' + ext_escape + '$')
            _get_prefix_from_saved_path_xdict[ext] = regexp

    matcher = regexp.match(path)
    if matcher is None:
        if force:
            raise Exception(f'Path "{path}" is not right!')
        else:
            return path
    prefix = matcher.group(1)
    return prefix


def get_path_from_prefix(xprefix, ext):
    """ Example: ext=.pth, "xxx/yyy" => "xxx/yyy.pth" """
    if xprefix is None:
        return None
    if ext == '':
        return xprefix
    return f'{xprefix}{ext}'


def get_tmp_file_name(path, new_ext=None, tgt_dir=None):
    """
    Generate a temporary file name.

    Example: path=/xxx/yyy/zzz.jpg new_ext=txt tgt_dir=/tmp => /tmp/zzz.txt
    :param path: The path to refer to.
    :param new_ext: The extension name of the temporary file name. If it is None, that will be the extension of "path".
    :param tgt_dir: The direction of the temporary file name. If it is None, that will be the CWD.
    :return:
    """
    arr = os.path.split(path)
    if tgt_dir is None:
        dir = '.'
    else:
        dir = tgt_dir
    name = arr[1]
    arr2 = os.path.splitext(name)
    base = arr2[0]
    if new_ext is None:
        ext = arr2[1]
    else:
        ext = new_ext
    os.makedirs(dir, exist_ok=True)
    result = os.path.join(dir, base + '.tmp.' + ext)
    return result


regexp4filename = re.compile(r'[^a-zA-Z0-9_\-.,;]')  # Go along with function ensure_filename
regexp4filename_no_dash_dot = re.compile(r'[^a-zA-Z0-9_,;]')  # Go along with function ensure_filename


def ensure_filename(name, no_dash_dot=False):
    if no_dash_dot:
        rexexp = regexp4filename_no_dash_dot
    else:
        rexexp = regexp4filename
    name = rexexp.sub('_', name)
    return name


shrink_underscore_regexp = re.compile(r'_+')


def shrink_underscore(input):
    return shrink_underscore_regexp.sub('_', input)


def rand_name_on_now():
    return shrink_underscore(ensure_filename(str(datetime.datetime.now()), no_dash_dot=True))


def check_np(arr, name):
    print(f'{name}: dtype: {arr.dtype}, shape: {arr.shape}, nbytes: {arr.nbytes}')


def check_np_detailed(arr, name):
    print(f'min: {arr.min():.2e},\tmax: {arr.max():.2e},\tmean: {arr.mean():.2e},\tstd: {arr.std():.2e},\tdtype: {arr.dtype},\tshape: {arr.shape},\tnbytes: {arr.nbytes},\t{name}')


def rand_color(low=0, high=256):
    return (
        np.random.randint(low, high),
        np.random.randint(low, high),
        np.random.randint(low, high),
    )


def rand_palette(n):
    pallette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1], dtype=int)
    colors = np.array([pallette * i % 255 for i in range(n)], dtype=np.uint8)
    return colors


if '__main__' == __name__:

    def _main():
        print(os.environ['PYTHONPATH'])

        """Unit tests for some routines declared in this file."""
        # ext = '.pth'
        ext = '/saved_model.pb'
        xpaths = [
            '/aaa/bbb/ccc/weights.100',
            '/aaa/bbb/ccc/weights.100/saved_model.pb',
            '/aaa/bbb/ccc/weights.100.pth',
            '/aaa/bbb/ccc/weights.100.index',
        ]
        for xpath in xpaths:
            sep()
            xprefix = get_prefix_from_path(xpath, ext, False)
            print(xpath, '=>', xprefix)
            xpath = get_path_from_prefix(xprefix, ext)
            print(xprefix, '=>', xpath)

    _main()
