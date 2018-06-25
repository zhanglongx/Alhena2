# coding: utf-8

import requests

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
