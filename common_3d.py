import numpy as np
from PyCmpltrtok.common import int_floor, int_round


def render_base_planes(sigma, value=1000.0):
    nx, ny, nz = sigma.shape[:3]
    ix = nx // 32
    iy = ny // 16
    iz = nz // 8
    sigma[0:1, ::ix, ::ix] += value
    sigma[::iy, 0:1, ::iy] += value
    sigma[::iz, ::iz, 0:1] += value


def f2n(f, N):
    return int_round(f * N)


def n2f(n, N):
    return n / N


def f2s(f, s0, s1):
    return f * (s1 - s0) + s0


def f2sd(f, s0, s1):
    return f * (s1 - s0)


def fit_021(f):
    f = max(f, 0.)
    f = min(f, 1.)
    return f


def f2slice(f0, f1):
    f0fit = fit_021(f0)
    if f0fit < 0.5:
        left_cut = f0fit - f0
    else:
        left_cut = 1.
    f1fit = fit_021(f1)
    if f1fit < 0.5:
        right_cut = 0.
    else:
        right_cut = f1 - f1fit
    return left_cut, right_cut, f0fit, f1fit


def add_sphere(sigma, X, Y, Z, cx, cy, cz, r, value=1000., xyz_sparse=True):
    nx, ny, nz = sigma.shape[:3]

    sx0 = X.min()
    sx1 = X.max()
    sy0 = Y.min()
    sy1 = Y.max()
    sz0 = Z.min()
    sz1 = Z.max()
    scx = f2s(cx, sx0, sx1)
    scy = f2s(cy, sy0, sy1)
    scz = f2s(cz, sz0, sz1)
    sr = f2sd(r, sx0, sx1)

    left_cut_x, right_cut_x, x0, x1 = f2slice(cx - r, cx + r)
    sn_x0 = f2n(x0, nx)
    sn_x1 = f2n(x1, nx)

    left_cut_y, right_cut_y, y0, y1 = f2slice(cy - r, cy + r)
    sn_y0 = f2n(y0, ny)
    sn_y1 = f2n(y1, ny)

    left_cut_z, right_cut_z, z0, z1 = f2slice(cz - r, cz + r)
    sn_z0 = f2n(z0, nz)
    sn_z1 = f2n(z1, nz)

    if xyz_sparse:
        xxx, yyy, zzz = get_slice_of_sparse_xyz(sn_x0, sn_x1, sn_y0, sn_y1, sn_z0, sn_z1, X, Y, Z)
    else:
        xxx, yyy, zzz = X[sn_x0:sn_x1, sn_y0:sn_y1, sn_z0:sn_z1], \
                        Y[sn_x0:sn_x1, sn_y0:sn_y1, sn_z0:sn_z1], \
                        Z[sn_x0:sn_x1, sn_y0:sn_y1, sn_z0:sn_z1]

    u = (
        (xxx - scx) ** 2
        + (yyy - scy) ** 2
        + (zzz - scz) ** 2
    ) < sr ** 2
    u = u.astype(np.float32) * value
    sigma[sn_x0:sn_x1, sn_y0:sn_y1, sn_z0:sn_z1] += u


def limit_in_range(x, low, high):
    x = max(x, low)
    x = min(x, high)
    return x


def validate_slice_ns(sn_x0, sn_x1, sn_y0, sn_y1, sn_z0, sn_z1, X, Y, Z):
    sn_x0 = sn_x0 if sn_x0 >= 0 else X.shape[0] + sn_x0
    sn_x1 = sn_x1 if sn_x1 >= 0 else X.shape[0] + sn_x1
    sn_y0 = sn_y0 if sn_y0 >= 0 else Y.shape[1] + sn_y0
    sn_y1 = sn_y1 if sn_y1 >= 0 else Y.shape[1] + sn_y1
    sn_z0 = sn_z0 if sn_z0 >= 0 else Z.shape[2] + sn_z0
    sn_z1 = sn_z1 if sn_z1 >= 0 else Z.shape[2] + sn_z1

    sn_x0 = limit_in_range(sn_x0, 0, X.shape[0])
    sn_x1 = limit_in_range(sn_x1, 0, X.shape[0])
    sn_y0 = limit_in_range(sn_y0, 0, Y.shape[1])
    sn_y1 = limit_in_range(sn_y1, 0, Y.shape[1])
    sn_z0 = limit_in_range(sn_z0, 0, Z.shape[2])
    sn_z1 = limit_in_range(sn_z1, 0, Z.shape[2])

    return sn_x0, sn_x1, sn_y0, sn_y1, sn_z0, sn_z1


def get_slice_of_sparse_xyz(sn_x0, sn_x1, sn_y0, sn_y1, sn_z0, sn_z1, X, Y, Z):
    sn_x0, sn_x1, sn_y0, sn_y1, sn_z0, sn_z1 = validate_slice_ns(sn_x0, sn_x1, sn_y0, sn_y1, sn_z0, sn_z1, X, Y, Z)
    nx, ny, nz = sn_x1 - sn_x0, sn_y1 - sn_y0, sn_z1 - sn_z0
    return np.broadcast_to(X[sn_x0:sn_x1, :, :], [nx, ny, nz]), \
           np.broadcast_to(Y[:, sn_y0:sn_y1, :], [nx, ny, nz]), \
           np.broadcast_to(Z[:, :, sn_z0:sn_z1], [nx, ny, nz])


def render_landmark_for_xyz(sigma, X, Y, Z, r=0.03, t=0.075, xyz_sparse=True):
    add_sphere(sigma, X, Y, Z, t, 0., 0., r, xyz_sparse=xyz_sparse)
    add_sphere(sigma, X, Y, Z, 0., t, 0., r, xyz_sparse=xyz_sparse)
    add_sphere(sigma, X, Y, Z, 0., 2. * t, 0., r, xyz_sparse=xyz_sparse)
    add_sphere(sigma, X, Y, Z, 0., 0., t, r, xyz_sparse=xyz_sparse)
    add_sphere(sigma, X, Y, Z, 0., 0., 2. * t, r, xyz_sparse=xyz_sparse)
    add_sphere(sigma, X, Y, Z, 0., 0., 3. * t, r, xyz_sparse=xyz_sparse)


def take_trunk_by_scale(ranges, s0, s1, is_cube=True):
    x1, x2, y1, y2, z1, z2 = ranges
    tx1 = f2s(x1, s0, s1)
    tx2 = f2s(x2, s0, s1)

    ty1 = f2s(y1, s0, s1)
    ty2 = f2s(y2, s0, s1)

    tz1 = f2s(z1, s0, s1)
    tz2 = f2s(z2, s0, s1)

    if is_cube:
        txx = tx2 - tx1
        tyy = ty2 - ty1
        tzz = tz2 - tz1
        tmax = max(txx, tyy, tzz)
        tx2 = tx1 + tmax
        ty2 = ty1 + tmax
        tz2 = tz1 + tmax
    return tx1, tx2, ty1, ty2, tz1, tz2


def get_xyz(nx, ny, nz, X, Y, Z):
    return (X[nx, 0, 0], Y[0, ny, 0], Z[0, 0, nz])


def get_chunk(ichunk, chunk_size, X, Y, Z, N):
    start = ichunk * chunk_size
    end = min((ichunk + 1) * chunk_size, N ** 3)
    res = np.zeros([end - start, 3], dtype=np.float32)
    for i, icoord in enumerate(range(start, end)):
        res[i] = get_xyz(icoord // (N*N), icoord // N % N, icoord % N, X, Y, Z)
    return res
