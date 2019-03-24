# coding: utf-8

import requests

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

    while(1):
        (text, raw) = _get_one_url(url, retries=-1, encoding=encoding)

        break

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
            resp = requests.get(url=url, headers=headers, timeout=60)
        except:
            continue

        if not resp.status_code == 200:
            continue

        resp.encoding = encoding

        # TODO: more check?
        break

    return (resp.text, resp.content)
