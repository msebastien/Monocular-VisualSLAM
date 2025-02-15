#!/usr/bin/env bash
# Written by Sébastien Maes

cd $HOME
echo "Executing g2o-python library install script"

# Clone g2o-python repo without its forked and outdated g2o git submodule
git clone --no-remote-submodules https://github.com/miquelmassot/g2o-python

# Clone 'pymem' branch from g2o github repo
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
git fetch --all
git checkout pymem
cd $HOME

# Copy all files from g2o github repo ('pymem' branch) to the g2o-python/g2o directory
cp -R -t g2o-python/g2o g2o/*

# Set the current directory to the Monocular-VisualSLAM repo
cd Monocular-VisualSLAM

# Checks if there is a python virtual environment and activate it
if [ -d .venv ]; then
    echo "A Python virtual environment has already been created. Activate it."
else
    echo "No Python virtual environment. A new one will be created."
    python3 -m venv .venv --system-site-packages
fi
source .venv/bin/activate

# Install g2o-python
python3 -m pip install -U -v $HOME/g2o-python/

# Clean git repos
rm -rf $HOME/g2o
rm -rf $HOME/g2o-python

echo "The g2o install script has completed its execution."