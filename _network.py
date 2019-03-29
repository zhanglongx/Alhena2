# coding: utf-8

import requests
import time

SLEEP_IN_S = 10

def _endless_get(url, param=None, encoding='utf-8'):
    '''
    endless get one url
    @params:
    url: {str}
    param: network param
        only None for now
    encoding: {str}
        return result in 'encoding'
    '''
    if not param is None:
        raise NotImplementedError

    (text, raw) = _get_one_url(url, retries=-1, encoding=encoding)

    return (text, raw)

def _get_one_url(url, retries=-1, encoding='utf-8'):
    '''
    Parameters
    ----------
    url: {str}
    retries: {int}
        timeout retries, -1 stands 'forever'
    encoding:
        return result in 'encoding'
    '''
    headers = {'User-Agent':'MyApp/1.0', 'Referer':'XXXXX'}

    # FIXME: support retries 
    while(1):
        try:
            resp = requests.get(url=url, headers=headers, timeout=SLEEP_IN_S)
        except:
            continue

        if not resp.status_code == 200:
            continue

        resp.encoding = encoding

        # TODO: more check?
        time.sleep(SLEEP_IN_S)
        break

    return (resp.text, resp.content)
