"""The plot style of the tutorials, shipped as ``timecorr/tutorial.mplstyle`` and used as
``plt.style.use('timecorr.tutorial')``. Real matplotlib; nothing is mocked."""

import subprocess
import sys
from importlib import resources

import matplotlib
import matplotlib.pyplot as plt

# what cell 3 of the quickstart, kernels, subjects, reduce, simulate, higher_order, events
# and decoding tutorials set before the style file existed
COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9', '#CC79A7', '#000000']
NOTEBOOK_RC = {
    'figure.dpi': 100, 'font.size': 11, 'axes.titlesize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.prop_cycle': plt.cycler(color=COLORS), 'legend.frameon': False,
}


def test_the_style_sets_exactly_what_the_notebooks_set():
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        plt.rcParams['figure.dpi'] = 72           # the style must set it, as the notebooks do
        # every setting but the backend's own (matplotlib changes those when a figure
        # manager is first needed)
        def settings():
            return {k: v for k, v in plt.rcParams.items() if not k.startswith('backend')}

        with plt.rc_context(NOTEBOOK_RC):
            expected = settings()
        with plt.style.context('timecorr.tutorial'):
            found = settings()
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        assert found == expected
        assert colors == COLORS
        assert plt.rcParams['figure.dpi'] == 72   # left as it was outside the context


def test_the_style_file_holds_the_seven_settings():
    text = resources.files('timecorr').joinpath('tutorial.mplstyle').read_text()
    keys = [line.split(':')[0].strip() for line in text.splitlines()
            if line.strip() and not line.startswith('#')]
    assert sorted(keys) == sorted(NOTEBOOK_RC)


def test_importing_timecorr_does_not_apply_the_style():
    keys = sorted(NOTEBOOK_RC)
    code = ("import matplotlib; before = dict(matplotlib.rcParams); import timecorr; "
            "import matplotlib.pyplot as plt; "
            f"print(all(plt.rcParams[k] == before[k] for k in {keys!r}), "
            "plt.rcParams['font.size'], plt.rcParams['axes.spines.top'])")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ['True', '10.0', 'True']
