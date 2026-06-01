"""Japanese font setup for matplotlib.

Import this module before any matplotlib/pyplot calls to enable
CJK (Japanese) character rendering using Noto Sans CJK JP.
Falls back silently if the font is not available.
"""

import matplotlib
import matplotlib.font_manager as fm

_JP_FONT_CANDIDATES = [
    'Noto Sans CJK JP',
    'Noto Serif CJK JP',
    'IPAexGothic',
    'IPAPGothic',
    'TakaoGothic',
    'VL Gothic',
]

_available = {f.name for f in fm.fontManager.ttflist}

for _candidate in _JP_FONT_CANDIDATES:
    if _candidate in _available:
        matplotlib.rcParams['font.family'] = _candidate
        break
