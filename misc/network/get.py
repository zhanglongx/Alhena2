# coding: utf-8

import requests
import re

def geturl(url, encoding='utf-8'):

    headers = {'User-Agent':'MyApp/1.0', 'Referer':'XXXXX'}

    # TODO: 
    while(1):
        try:
            resp = requests.get(url=url, headers=headers)
        except:
            continue

        if not resp.status_code == 200:
            continue

        resp.encoding = encoding

        # try parse the fisrt line
        text = resp.text
        first_line = text.split('\n', maxsplit=1)[0]

        r = re.compile('^报表日期\s+\d{8}\s+')
        if not r.match(first_line):
            continue

        break

        # TODO: more check

    return resp.content