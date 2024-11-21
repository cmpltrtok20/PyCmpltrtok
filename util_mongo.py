"""为聊天机器人项目量身定做的MongoDB函数集"""

import pymongo as pm
from PyCmpltrtok.common import translate_ns

KEY = 'key'
USER = 'user'
VALUE = 'value'
ID = '_id'

MONGODB_NAME = 'gradio_chatbot'

IO_PREFIX = {
    'i': '输入',
    'o': '输出',
}

# 索引定义
INDEXES = {
    # user_key: 唯一复合索引 USER（用户名）升序 + KEY（键，一般是纳秒时间戳）升序
    "user_key": {
        "cols": [
            (USER, pm.ASCENDING),
            (KEY, pm.ASCENDING),
        ],
        "uniq": True
    },
    # user_key: 唯一复合索引 USER（用户名）降序 + KEY（键，一般是纳秒时间戳）降序
    "user_key_desc": {
        "cols": [
            (USER, pm.DESCENDING),
            (KEY, pm.DESCENDING),
        ],
        "uniq": True
    },
    # gkey: 唯一单列索引 gkey升序（gkey = KEY + ',' + USER）
    "gkey": {
        "cols": [
            ('gkey', pm.ASCENDING),
        ],
        "uniq": True
    }
}


def setup_indexes(db, tbl_name, index_names):
    """
    建立索引，如果已经有了，则跳过。
    :param db: mongodb实例
    :param tbl_name: 表名
    :param index_names: 索引名字（必须是INDEXES里面指定的索引名。）
    :return: None
    """
    # 获取表
    tbl = db[tbl_name]

    # 列出已有索引
    indexes = set(tbl.index_information().keys())

    # 遍历index_names，如果没有这个索引，则按INDEXES里面的定义建立索引。
    for index_name in index_names:
        index_spec = INDEXES[index_name]
        if not index_name in indexes:
            tbl.create_index(
                index_spec["cols"],
                unique=index_spec["uniq"],
                name=index_name,
            )
            print(f'mongodb: Created index {index_name} in table {tbl_name}')


def mongo_upsert(db, tbl_name, user, key, value, index_names=('user_key', 'user_key_desc', 'gkey', )):
    """
    upsert = update or insert
    有数据则更新，无则插入。
    :param db: mongodb实例
    :param tbl_name: 表名
    :param user: 用户名
    :param key: 键（一般都是纳秒时间戳）
    :param value: 值 （如果是dict则按key-value插入值，如果不是dict则用键名'value'插入。）
    :param index_names: 要建立的索引名字（必须是INDEXES里面指定的索引名。）（如果已经存在则跳过。）
    :return: None
    """
    # 获取表
    tbl = db[tbl_name]

    # 组织要插入数据库的数据
    query = {
        USER: user,
        KEY: key,
    }
    data = query.copy()
    data['gkey'] = f'{key},{user}'
    if isinstance(value, dict):
        for k in value.keys():
            data[k] = value[k]
    else:
        data[VALUE] = value

    # upsert
    tbl.find_one_and_update(  # find_one_and_update = find_one + update
        query,
        {
            '$set': data,
        },
        upsert=True
    )

    # 处理索引（如果没有索引，则按index_names建立索引）
    setup_indexes(db, tbl_name, index_names)


def enqueue(db, tbl_name, user, key, value, index_names=('user_key', 'user_key_desc', 'gkey', )):
    """
    入队
    :param db: mongodb实例
    :param tbl_name: 表名
    :param user: 用户名
    :param key: 键（除了配置表conf，一般都是纳秒时间戳）
    :param value: 值 （如果是dict则按key-value插入值，如果不是dict则用键名'value'插入。）
    :param index_names: 要建立的索引名字（必须是global_conf["indexes"]里面指定的索引名。）（如果已经存在则跳过。）
    :return: None
    """
    mongo_upsert(db, tbl_name, user, key, value, index_names)


def dequeue(db, tbl_name, user=None, is_just_peek=True, order=pm.ASCENDING):
    """
    出队
    :param db: mongodb实例
    :param tbl_name: 表名
    :param user: 用户名
    :param is_just_peek: 是否获取后仍保留在队列里。
    :param order: 获取最早还是最晚
    :return: dict
    """
    # 获取表
    tbl = db[tbl_name]

    # 组织查询条件
    if user is not None:
        query = {
            USER: user
        }
        sort = [
            (USER, order),
            (KEY, order),
        ]
    else:
        query = {}
        sort = [
            ('gkey', order)
        ]

    if is_just_peek:
        # 只是读取
        row = tbl.find_one(query, sort=sort)
    else:
        # 出队（弹出） = 读取 + 移除
        row = tbl.find_one_and_delete(query, sort=sort)
    return row


def get_sorted_by_key(db, tbl_name, user, limit=None, is_keep_others=True, return_cursor=False):
    """
    按倒序获取一定量的数据
    主要使用场景：获取最近的一些聊天记录。
    :param db: mongodb实例
    :param tbl_name: 表名
    :param user: 用户名
    :param limit: 指定出队几条，为None时不做限制。
    :param is_keep_others: 其他数据是否保留
    :param return_cursor: 是否返回cursor
    :return: cursor or list of dict
    """
    # 获取表
    tbl = db[tbl_name]

    # 查询
    query = {
        USER: user,
    }
    cursor = tbl.find(query).sort([
        (USER, pm.DESCENDING),
        (KEY, pm.DESCENDING),
    ])

    # 如果只是返回游标
    if return_cursor:
        return cursor

    # 返回具体数据
    results = []
    border_ts = None
    cnt = -1
    for row in cursor:
        cnt += 1
        if limit is not None and cnt >= limit:
            break
        border_ts = row[KEY]
        results.append(row)

    # 当调用方指定删除查询范围外数据
    if len(results) and not is_keep_others and limit is not None and border_ts is not None:
        tbl.delete_many({
            USER: user,
            KEY: {
                '$lt': border_ts
            }
        })

    return results


def delete_many_by_user(db, tbl_name, user, criteria={}):
    """
    按用户，按条件批量删除。
    :param db: mongodb实例
    :param tbl_name: 表名
    :param user: 用户名
    :param criteria: 条件
    :return: None
    """
    # 获取表
    tbl = db[tbl_name]

    # 组织过滤条件
    criteria[USER] = user

    # 按条件删除数据
    tbl.delete_many(criteria)


def merge_dialog_in_and_out(rows_in, rows_out):
    """按时间倒序把聊天历史的输入和输出合并到一起。"""
    rows = rows_in + rows_out
    rows = sorted(rows, key=lambda row: row[KEY], reverse=True)  # DESC
    return rows


def get_history(mdb, username, more_info=False, limit=6, no_none=False):
    """
    获取聊天历史

    把历史组合成对子的列表，例如：
    in: xxx
    out: yyy
    in: zzzz
    in: aaaa
    out: bbbb
    out: kkkk
    =>
    [
        [xxx, yyy],
        [zzzz, None],
        [aaaa, bbbb],
        [None, kkkk],
    ]

    """
    assert(isinstance(limit, int), limit > 0)
    
    # 获取输入历史
    rows_in = get_sorted_by_key(mdb, 'dialog_in', username, limit=limit, is_keep_others=False)
    for xrow in rows_in:
        xrow['io'] = 'i'
    # 获取输出历史
    rows_out = get_sorted_by_key(mdb, 'dialog_out', username, limit=limit, is_keep_others=False)
    for xrow in rows_out:
        xrow['io'] = 'o'
    # 合并输入历史和输出历史
    rows = merge_dialog_in_and_out(rows_in, rows_out)  # DESC
    # 按时间正序排列
    rows = rows[::-1]  # ASC

    if not len(rows):
        # 没有历史则返回空列表
        log = []
    else:
        # 把聊天历史组合成对子的列表
        log = []
        pair = [None, None]
        for row in rows:
            text = row[VALUE]
            
            if not more_info:
                xstr = text
            else:
                timestamp = row[KEY]
                dt_str = translate_ns(timestamp)
                xstr = IO_PREFIX.get(row["io"], '?') + ': '
                xstr += f'[ {dt_str} ] '

                xstr += text

            if 'i' == row['io']:
                # 如果是输入

                # 如果对子里面已经有输入或输出了，则新建对子
                if pair[0] is not None or pair[1] is not None:
                    log.append(pair)
                    pair = [xstr, None]
                    continue

                # 输入放入对子的第一个元素
                pair[0] = xstr

            elif 'o' == row['io']:
                # 如果是输出

                # 如果对子里面已经有输出了，则新建对子
                if pair[1] is not None:
                    log.append(pair)
                    pair = [None, xstr]
                    continue

                # 输出放入对子的第2个元素
                pair[1] = xstr
        log.append(pair)

    if no_none:
        log = [[a if a is not None else '', b if b is not None else ''] for a, b in log]

    return log


def mongo_get(db, tbl_name, user, key, value=None, only_value=True, is_keep=True):
    """
    获取值（单条）
    :param db: mongodb实例
    :param tbl_name: 表名
    :param user: 用户名
    :param key: 键（除了配置表conf，一般都是纳秒时间戳）
    :param value: 默认值
    :param only_value: 为True则只返回'value'字段，为False则返回代表整行的dict
    :param is_keep: 获取后是否保留。为False时，获取后会删除这个数据。
    :return: dict or value
    """
    tbl = db[tbl_name]
    query = {
        USER: user,
        KEY: key,
    }
    row = tbl.find_one(query)
    if row is None:
        return value

    if only_value:
        value = row.get(VALUE, value)
    else:
        value = row

    if not is_keep:
        tbl.delete_one({ID: row[ID]})

    return value
