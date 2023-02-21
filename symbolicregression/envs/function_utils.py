# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, Tuple, List, Set, Dict
from pathlib import Path
import numpy as np

def zip_dic(lst):
    dico = {}
    for d in lst:
        if d is None:
            break
        for k in d:
            if k not in dico:
                dico[k] = []
            dico[k].append(d[k])
    for k in dico:
        if isinstance(dico[k][0], dict):
            dico[k] = zip_dic(dico[k])
    if dico == {}:
        return None
    return dico


def unsqueeze_dic(dico):
    dico_copy = {}
    for d in dico:
        if isinstance(dico[d], dict):
            dico_copy[d] = unsqueeze_dic(dico[d])
        else:
            dico_copy[d] = [dico[d]]
    return dico_copy


def squeeze_dic(dico):
    dico_copy = {}
    for d in dico:
        if isinstance(dico[d], dict):
            dico_copy[d] = squeeze_dic(dico[d])
        else:
            dico_copy[d] = dico[d][0]
    return dico_copy


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def getSizeOfNestedList(listOfElem):
    """Get number of elements in a nested list"""
    count = 0
    # Iterate over the list
    for elem in listOfElem:
        # Check if type of element is list
        if type(elem) == list:
            # Again call this function to get the size of this element
            count += getSizeOfNestedList(elem)
        else:
            count += 1
    return count


class ZMQNotReady(Exception):
    pass


class ZMQNotReadySample:
    pass

class TrainReader:
    def __init__(
        self,
        paths: List[Path],
        rng: np.random.RandomState,
        buffer_size: Optional[int] = 1,
        shuffle: Optional[bool] = True,
        keep_only_start: Optional[bool] = False,
        start: Optional[int] = 0,
        step: Optional[int] = 1,
        debug=False,
    ) -> None:

        self.paths = paths
        self.rng = rng
        self.curr_file_idx = self.rng.choice(range(len(paths)))
        self.file = open(self.paths[self.curr_file_idx], "r")
        self.keep_only_start = keep_only_start
        self.start = start
        self.step = step
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.buffer = []
        self.debug = debug
        for _ in range(self.start):
            self.file.readline()
        self.fill_buffer()
        assert len(self.buffer) == self.buffer_size
        if self.debug:
            print(
                f"Created TrainReader with buffer_size: {buffer_size}, start: {start}, step: {step} \n"
                f"Current path is {self.file.name}"
            )

    def is_file_empty(self, line) -> bool:
        if len(line) == 0:
            self.file.close()
            self.curr_file_idx = (self.curr_file_idx + 1) % len(self.paths)
            path = self.paths[self.curr_file_idx]
            if self.debug:
                print(
                    f"TrainReader reached the end of the file {self.file.name}, "
                    f"TrainReader opens file {path} ..."
                )
            self.file = open(path, "r")
            for _ in range(self.start):
                self.file.readline()
            return True
        return False

    def read_line(self) -> str:
        line = self.file.readline()
        while self.is_file_empty(line):
            line = self.file.readline()
        for _ in range(self.step - 1):
            line_skip = self.file.readline()
            if self.is_file_empty(line_skip):
                break
        return line

    def fill_buffer(self) -> None:
        while len(self.buffer) < self.buffer_size:
            line = self.read_line()
            if self.keep_only_start:
                if '"node": "0"' in line:
                    self.buffer.append(line)
            else:
                self.buffer.append(line)

    def __next__(self) -> str:
        line = self.buffer.pop(
            self.rng.randint(self.buffer_size) if self.shuffle else -1
        )
        self.fill_buffer()
        return line