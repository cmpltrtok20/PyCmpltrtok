"""
Hence the name "common redis", it is mainly for storing data to or retrieving data from Numpy.

https://stackoverflow.com/questions/55311399/fastest-way-to-store-a-numpy-array-in-redis
"""
import struct
import numpy as np


# https://docs.python.org/3/library/struct.html
# https://docs.python.org/3/library/struct.html#struct-format-strings
# > big-endian
# I unsigned int (4 byte)


def toRedis(r, a, n):
    """Store given Numpy array 'a' in Redis under key 'n' (2d array only) """
    h, w = a.shape
    shape = struct.pack('>II', h, w)
    encoded = shape + a.tobytes()

    # Store encoded data in Redis
    r.set(n, encoded)
    return


def fromRedis(r, n):
    """Retrieve Numpy array from Redis key 'n' (2d array only)"""
    encoded = r.get(n)
    h, w = struct.unpack('>II', encoded[:8])
    # Add slicing here, or else the array would differ from the original
    a = np.frombuffer(encoded[8:]).reshape(h, w)
    return a


def toRedisNd(r, a, n):
    """Store given Numpy array 'a' in Redis under key 'n' (nd array available, but less than 512M)"""
    len_shape = len(a.shape)
    nd = struct.pack('>I', len_shape)

    dtype = f'{a.dtype}'
    len_dtype = len(dtype)
    n_dtype = struct.pack('>I', len_dtype)

    shape_fmt = ''.join(['I' for i in range(len_shape)])
    shape = struct.pack('>' + shape_fmt, *a.shape)

    dtype_fmt = f'<{len_dtype}s'
    dtype_enc = struct.pack(dtype_fmt, dtype.encode(encoding='ascii'))

    encoded = nd + n_dtype + shape + dtype_enc + a.tobytes()

    # Store encoded data in Redis
    r.set(n, encoded)
    return


def fromRedisNd(r, n):
    """Retrieve Numpy array from Redis key 'n' (nd array available, but less than 512M)"""
    encoded = r.get(n)

    len_shape = struct.unpack('>I', encoded[:4])[0]
    len_dtype = struct.unpack('>I', encoded[4:8])[0]

    shape_fmt = ''.join(['I' for i in range(len_shape)])
    shape = struct.unpack('>' + shape_fmt, encoded[8:4*(2 + len_shape)])

    dtype_fmt = f'<{len_dtype}s'
    dtype = struct.unpack(dtype_fmt, encoded[4*(2 + len_shape):4*(2 + len_shape) + len_dtype])[0].decode(encoding='ascii')

    # Add slicing here, or else the array would differ from the original
    a = np.frombuffer(encoded[4*(2 + len_shape) + len_dtype:], dtype=dtype).reshape(*shape)
    return a


n512M = 2 ** 20 * 512


def toRedisNdLg(r, a, n):
    """Store given Numpy array 'a' in Redis under key 'n' (nd array available, any size)"""
    len_shape = len(a.shape)
    nd = struct.pack('>I', len_shape)

    dtype = f'{a.dtype}'
    len_dtype = len(dtype)
    n_dtype = struct.pack('>I', len_dtype)

    shape_fmt = ''.join(['I' for i in range(len_shape)])
    shape = struct.pack('>' + shape_fmt, *a.shape)

    dtype_fmt = f'<{len_dtype}s'
    dtype_enc = struct.pack(dtype_fmt, dtype.encode(encoding='ascii'))

    encoded = nd + n_dtype + shape + dtype_enc + a.tobytes()
    xlen = len(encoded)

    if xlen <= n512M:
        # Store encoded data in Redis
        r.set(n, encoded)
        return

    n_slice = int(np.ceil(xlen / n512M))
    for i_slice in range(n_slice):
        xslice = encoded[i_slice * n512M:(i_slice + 1) * n512M]
        if 0 == i_slice:
            r.set(n, xslice)
        else:
            r.set(f'{n}::::{i_slice:06d}', xslice)
    return


def fromRedisNdLg(r, n):
    """Retrieve Numpy array from Redis key 'n' (nd array available, any size)"""
    encoded = r.get(n)

    len_shape = struct.unpack('>I', encoded[:4])[0]
    len_dtype = struct.unpack('>I', encoded[4:8])[0]

    shape_fmt = ''.join(['I' for i in range(len_shape)])
    shape = struct.unpack('>' + shape_fmt, encoded[8:4*(2 + len_shape)])

    dtype_fmt = f'<{len_dtype}s'
    dtype = struct.unpack(dtype_fmt, encoded[4*(2 + len_shape):4*(2 + len_shape) + len_dtype])[0].decode(encoding='ascii')

    slice_arr = [encoded[4*(2 + len_shape) + len_dtype:]]
    i_slice = 1
    while True:
        key = f'{n}::::{i_slice:06d}'
        xslice = r.get(key)
        if xslice is None:
            break
        slice_arr.append(xslice)
        i_slice += 1

    xall = b''.join(slice_arr)
    a = np.frombuffer(xall, dtype=dtype).reshape(*shape)
    return a
