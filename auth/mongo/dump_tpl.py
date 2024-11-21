import yaml
from collections import OrderedDict

if '__main__' == __name__:
    
    xdict = {
        'host': 'localhost',
        'port': 27017,
        'username': 'root',
        'passwd': 'passwd',
        'None': None,
    }
    xdict = OrderedDict(xdict)
    
    path = 'mongodb.tpl.yaml'
    print(f'Dumping to {path}')
    with open(path, 'w', encoding='utf8') as f:
        yaml.dump(xdict, f, indent=4)
    print('Dumped')
    