# coding

import matplotlib.pyplot as plt

import Alhena2.cn.cn_extractor as ex

def plot(symbols, save_csv=False, start='2015-12-31', asfreq='A-DEC'):

    sub = ['PB', 'PE', 'ROE', 'CASH']

    data = ex.cn_extractor('.', symbols=symbols, subjects=sub, add_group='industry').gen_data()
    
    data = data.loc[start:].asfreq(asfreq)

    if save_csv == True:

        data.to_csv('t.csv', encoding='gb2312') 
        return

    plt.figure()

    for s in sub:

        data[s].plot.bar(title=s)

    plt.show()

def main():

    plot(['000651'], save_csv=True, start='2014-12-31', asfreq='A-DEC')

if __name__ == '__main__':
    main()