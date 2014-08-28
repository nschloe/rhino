#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Simple YAML emitter.
'''

# Python 2, 3 compatibility for print().
from __future__ import print_function


class YamlEmitter(object):
    '''
    Simple YAML emitter.
    '''
    def __init__(self):
        '''Initialization.
        '''
        self.envs = []
        self.indent = 0
        self.next_indent = self.indent
        self.key_is_next = True
        return

    def begin_doc(self):
        print('---')
        return

    def add_comment(self, comment):
        print(self.indent*' ' + '# ' + comment)
        return

    def begin_seq(self):
        if not self.envs:
            pass
        elif self.envs[-1] == 'seq':
            self.indent += 2
            self.next_indent = 0
            print('-', end=' ')
        elif self.envs[-1] == 'map':
            self.indent += 4
            self.next_indent = self.indent
            # print()
            self.key_is_next = True
        else:
            raise ValueError('Unknown environment.')
        self.envs.append('seq')
        return

    def add_item(self, item):
        assert self.envs
        assert self.envs[-1] == 'seq'
        print(self.next_indent*' ' + '-' + item)
        self.next_indent = self.indent
        return

    def end_seq(self):
        assert self.envs
        assert self.envs[-1] == 'seq'
        self.envs.pop()
        if not self.envs:
            pass
        elif self.envs[-1] == 'seq':
            self.indent -= 2
        elif self.envs[-1] == 'map':
            self.indent -= 4
        else:
            raise ValueError('Unknown environment.')
        self.next_indent = self.indent
        return

    def begin_map(self):
        if not self.envs:
            pass
        elif self.envs[-1] == 'seq':
            print(self.indent*' ' + '-', end=' ')
            self.indent += 2
            self.next_indent = 0
        elif self.envs[-1] == 'map':
            self.indent += 4
            self.next_indent = self.indent
            print()
        else:
            raise ValueError('Unknown environment.')
        self.envs.append('map')
        self.key_is_next = True
        return

    def add_key(self, key):
        assert self.envs
        assert self.envs[-1] == 'map'
        assert self.key_is_next
        print(self.next_indent*' ' + '%r:' % key)
        self.key_is_next = False
        return

    def add_value(self, item):
        assert self.envs
        assert self.envs[-1] == 'map'
        assert not self.key_is_next
        print('%r' % item)
        self.key_is_next = True
        self.next_indent = self.indent
        return

    def add_key_value(self, key, value):
        assert self.envs
        assert self.envs[-1] == 'map'
        assert self.key_is_next
        print(self.next_indent*' ' + '%r: %r' % (key, value))
        self.next_indent = self.indent
        return

    def end_map(self):
        assert self.envs
        assert self.envs[-1] == 'map'
        self.envs.pop()
        if not self.envs:
            pass
        elif self.envs[-1] == 'seq':
            self.indent -= 2
        elif self.envs[-1] == 'map':
            self.indent -= 4
        else:
            raise ValueError('Unknown environment.')
        self.next_indent = self.indent
        return
