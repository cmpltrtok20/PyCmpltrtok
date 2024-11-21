import configparser


if '__main__' == __name__:
    
    xdict = {
        'host': 'localhost',
        'port': '6379',
        'passwd': 'passwd',
        # 'None': None,  # TypeError: option values must be string
    }
    config = configparser.ConfigParser()
    for k, v in xdict.items():
        config['DEFAULT'][k] = v
    
    path = 'redis.tpl.ini'
    print(f'Dumping to {path}')
    with open(path, 'w', encoding='utf8') as f:
        config.write(f)
    print('Dumped')