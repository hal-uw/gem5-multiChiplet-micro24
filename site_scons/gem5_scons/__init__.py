# Copyright (c) 2013, 2015-2017 ARM Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Copyright (c) 2011 Advanced Micro Devices, Inc.
# Copyright (c) 2009 The Hewlett-Packard Development Company
# Copyright (c) 2004-2005 The Regents of The University of Michigan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import textwrap

from gem5_scons.util import get_termcap
import SCons.Script

termcap = get_termcap()

def strip_build_path(path, env):
    path = str(path)
    build_base = 'build/'
    variant_base = env['BUILDROOT'] + os.path.sep
    if path.startswith(variant_base):
        path = path[len(variant_base):]
    elif path.startswith(build_base):
        path = path[len(build_base):]
    return path

# Generate a string of the form:
#   common/path/prefix/src1, src2 -> tgt1, tgt2
# to print while building.
class Transform(object):
    # all specific color settings should be here and nowhere else
    tool_color = termcap.Normal
    pfx_color = termcap.Yellow
    srcs_color = termcap.Yellow + termcap.Bold
    arrow_color = termcap.Blue + termcap.Bold
    tgts_color = termcap.Yellow + termcap.Bold

    def __init__(self, tool, max_sources=99):
        self.format = self.tool_color + (" [%8s] " % tool) \
                      + self.pfx_color + "%s" \
                      + self.srcs_color + "%s" \
                      + self.arrow_color + " -> " \
                      + self.tgts_color + "%s" \
                      + termcap.Normal
        self.max_sources = max_sources

    def __call__(self, target, source, env, for_signature=None):
        # truncate source list according to max_sources param
        source = source[0:self.max_sources]
        def strip(f):
            return strip_build_path(str(f), env)
        if len(source) > 0:
            srcs = list(map(strip, source))
        else:
            srcs = ['']
        tgts = list(map(strip, target))
        # surprisingly, os.path.commonprefix is a dumb char-by-char string
        # operation that has nothing to do with paths.
        com_pfx = os.path.commonprefix(srcs + tgts)
        com_pfx_len = len(com_pfx)
        if com_pfx:
            # do some cleanup and sanity checking on common prefix
            if com_pfx[-1] == ".":
                # prefix matches all but file extension: ok
                # back up one to change 'foo.cc -> o' to 'foo.cc -> .o'
                com_pfx = com_pfx[0:-1]
            elif com_pfx[-1] == "/":
                # common prefix is directory path: OK
                pass
            else:
                src0_len = len(srcs[0])
                tgt0_len = len(tgts[0])
                if src0_len == com_pfx_len:
                    # source is a substring of target, OK
                    pass
                elif tgt0_len == com_pfx_len:
                    # target is a substring of source, need to back up to
                    # avoid empty string on RHS of arrow
                    sep_idx = com_pfx.rfind(".")
                    if sep_idx != -1:
                        com_pfx = com_pfx[0:sep_idx]
                    else:
                        com_pfx = ''
                elif src0_len > com_pfx_len and srcs[0][com_pfx_len] == ".":
                    # still splitting at file extension: ok
                    pass
                else:
                    # probably a fluke; ignore it
                    com_pfx = ''
        # recalculate length in case com_pfx was modified
        com_pfx_len = len(com_pfx)
        def fmt(files):
            f = list(map(lambda s: s[com_pfx_len:], files))
            return ', '.join(f)
        return self.format % (com_pfx, fmt(srcs), fmt(tgts))

# The width warning and error messages should be wrapped at.
text_width = None

# If stdout is not attached to a terminal, default to 80 columns.
if not sys.stdout.isatty():
    text_width = 80

# This should work in python 3.3 and above.
if text_width is None:
    try:
        import shutil
        text_width = shutil.get_terminal_size().columns
    except:
        pass

# This should work if the curses python module is installed.
if text_width is None:
    try:
        import curses
        try:
            _, text_width = curses.initscr().getmaxyx()
        finally:
            curses.endwin()
    except:
        pass

# If all else fails, default to 80 columns.
if text_width is None:
    text_width = 80

def print_message(prefix, color, message, **kwargs):
    prefix_len = len(prefix)
    if text_width > prefix_len:
        wrap_width = text_width - prefix_len
        padding = ' ' * prefix_len

        # First split on newlines.
        lines = message.split('\n')
        # Then wrap each line to the required width.
        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(textwrap.wrap(line, wrap_width))
        # Finally add the prefix and padding on extra lines, and glue it all
        # back together.
        message = prefix + ('\n' + padding).join(wrapped_lines)
    else:
        # We have very small terminal, indent formatting doesn't help.
        message = prefix + message
    # Add in terminal escape sequences.
    message = color + termcap.Bold + message + termcap.Normal
    # Actually print the message.
    print(message, **kwargs)
    return message

all_warnings = []
def summarize_warnings():
    if not all_warnings:
        return
    print(termcap.Yellow + termcap.Bold +
            '*** Summary of Warnings ***' +
            termcap.Normal)
    list(map(print, all_warnings))

def warning(*args, **kwargs):
    message = ' '.join(args)
    printed = print_message('Warning: ', termcap.Yellow, message, **kwargs)
    all_warnings.append(printed)

def error(*args, **kwargs):
    message = ' '.join(args)
    print_message('Error: ', termcap.Red, message, **kwargs)
    SCons.Script.Exit(1)

__all__ = ['Transform', 'warning', 'error']
