import pymongo as pm
import sys
import json
import collections as col

MAP_INDEX_NAME = 'key_of_map'


def save_map(xtable, xkey, xvalue):
    assert(isinstance(xtable, pm.collection.Collection))

    xkey_json = json.dumps(xkey)
    xvalue_json = json.dumps(xvalue)
    xfilter = {
        'key': xkey_json,
    }
    xdata = {
        **xfilter,
        'value': xvalue_json
    }
    xtable.replace_one(xfilter, xdata, True)

    indexes = set(xtable.index_information().keys())
    if not MAP_INDEX_NAME in indexes:
        xtable.create_index([
            ('key', pm.ASCENDING),
        ], unique=True, name=MAP_INDEX_NAME)
        print(f'> TVTS: Created index {MAP_INDEX_NAME} of {xtable}', file=sys.stderr)


def read_map(xtable, xkey):
    assert (isinstance(xtable, pm.collection.Collection))

    xkey_json = json.dumps(xkey)
    xrecord = xtable.find_one({
        'key': xkey_json,
    })
    if xrecord is None:
        return None
    xvalue_json = xrecord['value']
    xvalue = json.loads(xvalue_json)
    return xvalue


class PymongoDict(col.UserDict):

    def __init__(self, xdb, xtable, xhost='localhost', xport=27017):
        assert(isinstance(xdb, str))
        assert(isinstance(xtable, str))
        assert(isinstance(xhost, str))
        assert(isinstance(xport, int))

        client = pm.MongoClient(xhost, xport)
        self.table = client[xdb][xtable]

    def __setitem__(self, key, value):
        save_map(self.table, key, value)

    def __getitem__(self, key):
        value = read_map(self.table, key)
        return value


if '__main__' == __name__:
    xdict = PymongoDict('db4test', 'map4test')
    xdict_normal = {}
    for k in 'abcde':
        xdict_normal[k] = k * 4
        xdict[k] = k * 4
    for k in 'abcde':
        print(k, xdict_normal[k])
        print(k, xdict[k])
