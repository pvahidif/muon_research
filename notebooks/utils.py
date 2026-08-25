# pylint: disable=all

from pathlib import Path
import glob
import json
import os
import sys

from tqdm.auto import tqdm
from IPython.display import Image, display
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

pl.Config.set_tbl_rows(250)
pl.Config.set_tbl_cols(50)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_fmt_str_lengths(200)
