import yaml
import pymongo
import argparse
import copy
import os
from PyCmpltrtok.common import sep, uuid, get_dir_name_ext
from PyCmpltrtok.util_mongo import get_history, enqueue

print(f'__name__=|{__name__}|')


def parse(name):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir, f'mongodb.{name}.tmp.yaml')
    print(f'Loading from {path}')
    with open(path, 'r', encoding='utf8') as f:
        xobj = yaml.load(f, Loader=yaml.Loader)
    print('Loaded')
    
    xobj_print = copy.deepcopy(xobj)
    xobj_print['passwd'] = '****'
    sep()
    print(xobj_print)
    sep()
    return xobj


def conn(name='local'):
    xobj = parse(name)
    sep('Connecting MongoDB ....')
    args = [
        xobj['host'], int(xobj['port'])
    ]
    kwargs = {}
    for k, k2 in (('username', 'username', ), ('password', 'passwd', )):
        if xobj[k2] is not None:
            kwargs[k] = xobj[k2]
        
    mongo = pymongo.MongoClient(
        *args, 
        serverSelectionTimeoutMS=3000,
        # connect=False,
        **kwargs
    )
    mdb = mongo['test']
    get_history(mdb, 'user_xxxx', limit=1)  # try it
    sep('MongoDB OK')
    return mongo
    

if '__main__' == __name__:
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', help='name of the config', default='local')
    args = parser.parse_args()
    name = args.name
    mongo = conn(name)
    print('HOST:', mongo.HOST)
    print('PORT:', mongo.PORT)
    
    mdb = mongo['test001']
    xuuid = uuid()
    r = enqueue(mdb, 'tbl001', 'user001', xuuid, '测试数据001 - test data 001')
    print(r)
