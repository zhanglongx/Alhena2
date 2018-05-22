# coding: utf-8

import os
import pandas as pd


def loc_tickers(path):  
    loc = os.path.join(path, 'database', 'tickers')

    if not os.path.exists(loc):
        os.mkdir(loc)

    return loc

def file_tickers(path):
    return os.path.join(loc_tickers(path), 'tickers.csv')