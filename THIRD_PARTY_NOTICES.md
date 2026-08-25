# Third-Party Notices

This repository's training/architecture code lineage traces back to Keller
Jordan's [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
(the nanoGPT speedrun), which itself descends from Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and
[llm.c](https://github.com/karpathy/llm.c). The files below are adapted
directly from those projects and retain the applicable upstream MIT license
notice at the top of the file. This document reproduces those notices in one
place as well.

## Files adapted from `karpathy/llm.c`

- [`src/muon_research/download_data/fineweb.py`](src/muon_research/download_data/fineweb.py)
  — adapted from `dev/data/fineweb.py`.

```
MIT License

Copyright (c) 2024 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Files adapted from `KellerJordan/modded-nanogpt`

- [`src/muon_research/download_data/cached_fineweb10B.py`](src/muon_research/download_data/cached_fineweb10B.py)
  — adapted from `data/cached_fineweb10B.py`.
- [`src/muon_research/download_data/cached_fineweb100B.py`](src/muon_research/download_data/cached_fineweb100B.py)
  — adapted from `data/cached_fineweb100B.py`.
- [`src/muon_research/download_data/cached_finewebedu10B.py`](src/muon_research/download_data/cached_finewebedu10B.py)
  — adapted from `data/cached_finewebedu10B.py`.

Additionally, [`src/muon_research/download_data/cached_fineweb10B_vocab.py`](src/muon_research/download_data/cached_fineweb10B_vocab.py)
and [`src/muon_research/download_data/cached_fineweb10B_test.py`](src/muon_research/download_data/cached_fineweb10B_test.py)
are original code, but reuse the `.bin` shard header format and download
conventions established in `cached_fineweb10B.py` above.

More broadly, this repository's training loop, model architecture, and
experiment scaffolding are derived from and build on modded-nanogpt.

```
MIT License

Copyright (c) 2024 Keller Jordan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## This repository's own license

Original code in this repository (i.e. everything not called out above) is
licensed under the MIT license in [`LICENSE`](LICENSE), Copyright (c) 2026
Pooya Vahidi Ferdowsi. That license does not, by itself, relicense or
supersede the upstream notices reproduced above; it applies to this
repository's own contributions.
