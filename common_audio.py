import struct

# https://docs.python.org/3/library/struct.html
# https://docs.python.org/3/library/struct.html#struct-format-strings
# > big-endian
# I unsigned int (4 byte)

WAV_HEADER_LEN = 44


def compose_wav_header(bytes_len, num_channels=1, bit_size=16, sample_rate=16000):
    assert(num_channels in [1, 2], 'num_channels must be 1 or 2!')

    header_arr = []

    header_arr.append(b'RIFF')

    header_arr.append(struct.pack('<I', bytes_len + WAV_HEADER_LEN - 8))

    header_arr.append(b'WAVE')
    header_arr.append(b'fmt ')

    header_arr.append(struct.pack('<I', 16))

    header_arr.append(struct.pack('<H', 1))

    header_arr.append(struct.pack('<H', num_channels))

    header_arr.append(struct.pack('<I', sample_rate))

    bytes_per_sec = sample_rate * bit_size * num_channels // 8
    header_arr.append(struct.pack('<I', bytes_per_sec))

    block_alignment = bit_size * num_channels // 8
    header_arr.append(struct.pack('<H', block_alignment))

    header_arr.append(struct.pack('<H', bit_size))

    header_arr.append(b'data')

    header_arr.append(struct.pack('<I', bytes_len))

    header_bytes = b''.join(header_arr)

    assert(WAV_HEADER_LEN == len(header_bytes), f'Header size must be {WAV_HEADER_LEN} bytes!')

    return header_bytes


if '__main__' == __name__:
    from PyCmpltrtok.common import get_dir_name_ext, rand_name_on_now
    import os

    XDIR, XNAME, _ = get_dir_name_ext(os.path.abspath(__file__))
    XSAVE_DIR = os.path.join(XDIR, '_save', XNAME)
    os.makedirs(XSAVE_DIR, exist_ok=True)

    path = '_save/1681285791362293300.wav'
    print(f'Read from {path}')
    _, XNAME, _ = get_dir_name_ext(path)
    XSAVE_PATH = os.path.join(XSAVE_DIR, f'{XNAME}.{rand_name_on_now()}.wav')
    with open(path, 'br') as f:
        audio = f.read()
        head = audio[:44]
        print('head:', head)
        data = audio[44:]
        xlen = len(data)
        my_head = compose_wav_header(xlen)
        print('myhd:', my_head)
        with open(XSAVE_PATH, 'bw') as fw:
            fw.write(my_head)
            fw.write(data)
    print(f'Write to {XSAVE_PATH}')
