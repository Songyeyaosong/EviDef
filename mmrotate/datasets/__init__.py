# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .sodaa import SODAADataset
from .fair import FairDataset
from .dota2 import DOTA2Dataset
from .dota_open import DOTAOpenDataset

__all__ = ['SARDataset', 'DOTADataset', 'build_dataset', 'HRSCDataset', 'SODAADataset', 'FairDataset', 'DOTA2Dataset', 'DOTAOpenDataset']
