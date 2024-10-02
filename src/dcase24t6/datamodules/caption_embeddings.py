#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp
from pathlib import Path
from typing import Any, Callable, ClassVar, List, Optional, Union

from typing_extensions import NotRequired, TypedDict

from aac_datasets import Clotho
import torch

pylog = logging.getLogger(__name__)

NUMBER_OF_SUMMARIZATIONS = 3


class CaptionEmbeddings(Clotho):
    def __init__(
        self,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        print(f'Processing {self.subset} dataset')
        summarizations = []
        ids = [i for i in range(len(self))]
        self.add_raw_column('ids', ids)
        self._columns.append('ids')

        if self.subset in ['dev']:
            for i in range(1, NUMBER_OF_SUMMARIZATIONS + 1):
                with open(f'./data/summarizations/summarizations_{self.subset}_{i}.txt', 'r') as f:
                    for j, line in enumerate(f):
                        if len(summarizations) <= j:
                            summarizations.append([])
                        summarizations[j].append(line.replace('\n', ''))
            for i, item in enumerate(self):
                captions = item['captions']
                for j in range(NUMBER_OF_SUMMARIZATIONS):
                    captions.append(summarizations[i][j])


def main():
    CaptionEmbeddings()


if __name__ == "__main__":
    main()
