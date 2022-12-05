import os
from argparse import ArgumentParser
import argparse

from jutils.ngc_utils import wrap_cmd


def add_ngc_args(arg_parser: ArgumentParser):
    arg_parser.add_argument("--pid", type=str)
    return arg_parser


def kill_all():
    for pid in args.pid.split(','):
        wrap_cmd('ngc batch kill %s' % pid)


if __name__ == '__main__':
    parser = ArgumentParser()
    add_ngc_args(parser)
    args = parser.parse_args()
    kill_all()