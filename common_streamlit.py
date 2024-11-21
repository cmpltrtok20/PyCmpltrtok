from streamlit_cookies_controller import CookieController


def get_all_cookies():
    '''
    WARNING: This uses unsupported feature of Streamlit
    Returns the cookies as a dictionary of kv pairs
    '''
    from streamlit.web.server.websocket_headers import _get_websocket_headers 
    # https://github.com/streamlit/streamlit/pull/5457
    from urllib.parse import unquote

    headers = _get_websocket_headers()
    if headers is None:
        return {}
    
    if 'Cookie' not in headers:
        return {}
    
    cookie_string = headers['Cookie']
    # A sample cookie string: "K1=V1; K2=V2; K3=V3"
    cookie_kv_pairs = cookie_string.split(';')

    cookie_dict = {}
    for kv in cookie_kv_pairs:
        k_and_v = kv.split('=')
        k = k_and_v[0].strip()
        v = k_and_v[1].strip()
        cookie_dict[k] = unquote(v) #e.g. Convert name%40company.com to name@company.com
    return cookie_dict


def set_cookie(name, value, max_age=3600*24*30):
    controller = CookieController()
    controller.set(name, value, max_age=max_age)