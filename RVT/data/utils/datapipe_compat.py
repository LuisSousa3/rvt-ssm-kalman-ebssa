"""Compatibility imports for PyTorch DataPipes.

Older RVT code imports DataPipes from torchdata.datapipes.*. Newer torchdata
versions removed that namespace, while PyTorch still provides the same core
types under torch.utils.data.
"""

from __future__ import annotations

from itertools import zip_longest
from typing import Any

try:
    from torchdata.datapipes.iter import (
        Concater,
        IterableWrapper,
        IterDataPipe,
        Zipper,
        ZipperLongest,
    )
    from torchdata.datapipes.map import MapDataPipe
except ModuleNotFoundError:
    from torch.utils.data import IterDataPipe, MapDataPipe
    from torch.utils.data.datapipes.iter import Concater, IterableWrapper, Zipper

    class _CycleIterDataPipe(IterDataPipe):
        """Fallback adapter for IterDataPipe.cycle() on newer PyTorch releases."""

        def __init__(self, source_datapipe: IterDataPipe, count: int | None = None):
            super().__init__()
            self.source_datapipe = source_datapipe
            self.count = count

        def __iter__(self):
            if self.count is None:
                while True:
                    yielded = False
                    for item in self.source_datapipe:
                        yielded = True
                        yield item
                    if not yielded:
                        return
                return

            for _ in range(self.count):
                for item in self.source_datapipe:
                    yield item

    class _MapToIterDataPipe(IterDataPipe):
        """Fallback adapter for map-style datapipes on newer PyTorch releases."""

        def __init__(self, source_datapipe: MapDataPipe):
            super().__init__()
            self.source_datapipe = source_datapipe

        def __iter__(self):
            for index in range(len(self.source_datapipe)):
                yield self.source_datapipe[index]

        def __len__(self) -> int:
            return len(self.source_datapipe)

    if not hasattr(MapDataPipe, "to_iter_datapipe"):

        def _to_iter_datapipe(self) -> IterDataPipe:
            return _MapToIterDataPipe(self)

        MapDataPipe.to_iter_datapipe = _to_iter_datapipe

    if not hasattr(IterableWrapper([0]), "cycle"):

        def _cycle(self, count: int | None = None) -> IterDataPipe:
            return _CycleIterDataPipe(self, count=count)

        IterDataPipe.cycle = _cycle

    class ZipperLongest(IterDataPipe):
        """Small fallback for torchdata.datapipes.iter.ZipperLongest."""

        def __init__(self, *datapipes: IterDataPipe, fill_value: Any = None):
            super().__init__()
            self.datapipes = datapipes
            self.fill_value = fill_value

        def __iter__(self):
            yield from zip_longest(*self.datapipes, fillvalue=self.fill_value)

        def __len__(self) -> int:
            return max(len(datapipe) for datapipe in self.datapipes)
