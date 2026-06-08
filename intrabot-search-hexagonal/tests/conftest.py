"""
Root conftest.py — executed once before the test suite.

Sets sys.path so that `import app.*` and `import tests.*` both resolve
from the project root, whether pytest is launched with `pytest` or
`python -m pytest` from any working directory.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
