# coding: utf-8

import os, sys, argparse
from abc import (ABCMeta, abstractmethod)

from cn.cn_reader import (cn_reader)

def _progress_bar(inter, total):
    # copied from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    # Print iterations progress
    def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '>'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total:
            print()

    return printProgressBar(iteration=inter, total=total, prefix='Progress:', suffix='Complete', length=30)

class _base():
    __metaclass__ = ABCMeta

    def __init__(self, path):
        self.path = path

    def update(self):
        return self.__query_yes_no('update can take up a *VERY* long time. ', \
                                   default='no')

    @abstractmethod
    def build(self, file):
        raise NotImplementedError

    @staticmethod
    def __query_yes_no(question, default="yes"):
        '''
        https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
        Ask a yes/no question via input() and return their answer.

        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

        The "answer" return value is True for "yes" or False for "no".
        '''
        valid = {"yes": True, "y": True, "ye": True,
                 "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            # choice = input().lower()
            choice = 'yes' # tempz
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                 "(or 'y' or 'n').\n")

class cn(_base):
    def __init__(self, path):
        super().__init__(path)

        # self._reader = cn_reader(path, symbols='000001') # tempz
        self._reader = cn_reader(path, symbols=None) # tempz

    def update(self):
        if not super().update() is True:
            return False

        # self._reader.update(category='daily') # tempz
        self._reader.update(cb_progress=_progress_bar, category='daily')

        return True

    def build(self, file=None):
        self._reader.build(file=file)

def main():
    parser = argparse.ArgumentParser(description='wrapper for Alhena2 reader')
    parser.add_argument('-f', '--file', type=str, nargs='?', help='file to built')
    parser.add_argument('-m', '--market', default='cn', type=str, nargs='?', help='Alhena2 reader market')
    parser.add_argument('-p', '--path', default='.', type=str, nargs='?', help='Alhena2 path')
    parser.add_argument('command', type=str, nargs='+', help='command to execute, support commands: update build')

    market  = parser.parse_args().market
    file    = parser.parse_args().file
    path    = parser.parse_args().path
    command = parser.parse_args().command

    if market == 'cn':
        reader = cn(path)
    else:
        raise ValueError('market %s has not been implemented' % market)

    for c in command:
        if c == 'update':
            reader.update()
        elif c == 'build':
            reader.build(file)
        else:
            raise ValueError('command %s not supported' % c)

if __name__ == '__main__':
    main()