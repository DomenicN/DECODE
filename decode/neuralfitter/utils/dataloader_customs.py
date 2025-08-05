import torch
import torch.utils.data
from torch.utils.data._utils import collate
from types import NoneType
from typing import Callable, Optional, Union
import collections.abc as container_abcs

import decode.generic


def collate_nonetype_fn(
    batch,
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
):
    return None


def collate_emitterset_fn(
    batch,
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
):
    return [em for em in batch]


def smlm_collate(batch):
    """
    Collate for dataloader that allows for None return and EmitterSet.
    Otherwise defaults to default pytorch collate

    Args:
        batch
    """
    elem = batch[0]
    if elem is None:
        return None

    collate.default_collate_fn_map.update({NoneType: collate_nonetype_fn,
                                           decode.generic.emitter.EmitterSet: collate_emitterset_fn})
    return collate.default_collate(batch)
