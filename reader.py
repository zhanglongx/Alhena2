# coding: utf-8

import os, sys, argparse
from abc import (ABCMeta, abstractmethod)

from cn.cn_reader import (cn_reader)

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

        self._reader = cn_reader(path, ['000651', '600060']) # tempz

    def update(self):
        if not super().update() is True:
            return False

        self._reader.update()

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