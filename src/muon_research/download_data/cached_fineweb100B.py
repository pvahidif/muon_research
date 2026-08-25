"""
Adapted from modded-nanogpt (https://github.com/KellerJordan/modded-nanogpt),
data/cached_fineweb100B.py.

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

See THIRD_PARTY_NOTICES.md at the repo root for details.
"""

import os
import sys
from huggingface_hub import hf_hub_download

from muon_research.paths import REPO_ROOT


# Download the GPT-2 tokens of Fineweb100B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
def get(fname):
    local_dir = os.path.join(REPO_ROOT, "data", "fineweb100B")
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(
            repo_id="kjj0/fineweb100B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=local_dir,
        )


get("fineweb_val_%06d.bin" % 0)
num_chunks = 1030  # full fineweb100B. Each chunk is 100M tokens
if len(sys.argv) >= 2:  # we can pass an argument to download less
    num_chunks = int(sys.argv[1])
for i in range(1, num_chunks + 1):
    get("fineweb_train_%06d.bin" % i)
