import configparser
import redis
import argparse
import copy
import os
from PyCmpltrtok.common import sep, uuid, get_dir_name_ext
from PyCmpltrtok.util_mongo import get_history, enqueue

print(f'__name__=|{__name__}|')


def parse(name):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir, f'redis.{name}.tmp.ini')
    conf = configparser.ConfigParser()
    print(f'Loading from {path}')
    with open(path, 'r', encoding='utf8') as f:
        conf.read_file(f)
    print('Loaded')
    
    # https://stackoverflow.com/questions/1773793/convert-configparser-items-to-dictionary
    xobj = dict(conf.items('DEFAULT'))
    
    xobj_print = copy.deepcopy(xobj)
    xobj_print['passwd'] = '****'
    sep()
    print(xobj_print)
    sep()
    return xobj


def conn(name='local', db_idx=0, timeout_sec=3) -> redis.Redis:
    xobj = parse(name)
    sep('Connecting Redis ....')
    args = [
        xobj['host'], int(xobj['port'])
    ]
    kwargs = {}
    for k, k2 in (('password', 'passwd', ), ):
        if xobj[k2] is not None:
            kwargs[k] = xobj[k2]
        
    rdb = redis.Redis(
        *args, db_idx, socket_timeout=timeout_sec,
        **kwargs
    )
    rdb.get('try_it')
    sep('Redis OK')
    return rdb
    

if '__main__' == __name__:
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', help='name of the config', default='local')
    args = parser.parse_args()
    name = args.name
    rdb = conn(name)
    info = rdb.get_connection_kwargs()
    sep('Conn info')
    print('host', info['host'])
    print('port', info['port'])
    sep('Conn info over')
    
    xuuid = uuid()
    key = f'test_{xuuid}'
    data = '测试数据001 - test data 001'
    rdb.set(key, data.encode('utf8'))
    data02 = rdb.get(key).decode('utf8')
    rdb.delete(key)
    assert data == data02
    
    print('Test passed.')
