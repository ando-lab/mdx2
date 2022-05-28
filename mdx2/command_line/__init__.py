
#import os
#import argparse

# disable long tracebacks for command line programs
#sys.tracebacklimit = 0


# def output_file(arg):
#     from os.path import exists
#     if exists(arg):
#         print(f'output file {arg} already exists!')
#         #raise argparse.ArgumentTypeError('invalid value!!!')
#     return arg
#
# def input_file(arg):
#     from os.path import exists
#     if not exists(arg):
#         print(f'input file {arg} does not exist!')
#     return arg
#
# class MDX2Parser(argparse.ArgumentParser):
#     def __init__(self,*args,**kwargs):
#         if 'description' not in kwargs:
#             kwargs['description'] = 'MXD2 command line program'
#         if 'formatter_class' not in kwargs:
#             kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
#         super().__init__(self,*args,**kwargs)
#
#     def add_output_file_argument(self, *args, **kwargs):
#         kwargs['type'] = output_file
#         self.add_argument(*args, **kwargs)
#
#     def add_input_file_argument(self, *args, **kwargs):
#         kwargs['type'] = input_file
#         self.add_argument(*args, **kwargs)

# def raise_if_file_exists(*filenames):
#     """error if file exists"""
#
#     for filename in filenames:
#         if os.path.exists(filename):
#             raise FileExistsError(
#                 errno.EEXIST,
#                 os.strerror(errno.EEXIST),
#                 filename,
#                 )
#
# def raise_if_file_not_found(*filenames):
#     """error if file not found"""
#
#     for filename in filenames:
#         if not os.path.exists(filename):
#             raise FileNotFoundError(
#                 errno.ENOENT,
#                 os.strerror(errno.ENOENT),
#                 filename,
#                 )
