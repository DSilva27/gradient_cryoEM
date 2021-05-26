conda config --add channels conda-forge
conda update --yes conda
conda create --yes -n em2d_env python=3.8 jupyterlab matplotlib numpy
conda install --yes -n em2d_env MDAnalysis MDAnalysisTests
conda activate em2d_env