#!/usr/bin/env bash
# Written by Sébastien Maes

IS_VENV_INITIALIZED=false

# Checks if there is a python virtual environment and activate it
check_python_venv () {
    if [ -d "$1"/.venv ]; then
        echo -n "A Python virtual environment has already been created. Activate it."
    else
        echo -n "No Python virtual environment. A new one will be created."
        python3 -m venv .venv --system-site-packages
    fi

    if [[ "$IS_VENV_INITIALIZED" == false ]]; then
        source "$1"/.venv/bin/activate
        IS_VENV_INITIALIZED=true
    fi
}

install_g2o () {
    cd $HOME
    echo -n "==> Start running g2o-python library install script..."

    # Clone g2o-python repo without its forked and outdated g2o git submodule
    git clone --no-remote-submodules https://github.com/miquelmassot/g2o-python

    # Clone 'pymem' branch from g2o github repo
    git clone https://github.com/RainerKuemmerle/g2o.git
    cd g2o || exit
    git fetch --all
    git checkout pymem
    cd $HOME

    # Copy all files from g2o github repo ('pymem' branch) to the g2o-python/g2o directory
    cp -R -t g2o-python/g2o g2o/*

    # Set the current directory to the Monocular-VisualSLAM repo
    cd Monocular-VisualSLAM || exit

    check_python_venv "$(pwd)"

    # Install g2o-python
    python3 -m pip install -U -v "$HOME"/g2o-python/

    # Clean git repos
    rm -rf "$HOME"/g2o
    rm -rf "$HOME"/g2o-python

    echo -n "=========================================================================="
    echo -n "|   The g2o-python library install script has completed its execution.   |"
    echo -n "=========================================================================="
}

install_pangolin () {
    cd $HOME
    echo -n "==> Start running pangolin vizualisation library install script..."

    git clone https://github.com/uoip/pangolin.git
    
    # Set current directory to the cloned repo
    cd pangolin || exit

    # Replace setup file with the fixed one
    rm -f setup.py
    cp "$HOME"/Monocular-VisualSLAM/pangolin/setup.py .

    # Create build directory
    mkdir build

    # Generate the Makefile using CMake then
    # build the library using Make and the generated Makefile
    cd build
    cmake ..
    make -j8
    cd $HOME

    # Set the current directory to the Monocular-VisualSLAM repo
    cd Monocular-VisualSLAM || exit

    check_python_venv "$(pwd)"

    # Install library
    python3 -m pip install "$HOME"/pangolin

    # Clean git repo and build files
    rm -rf "$HOME"/pangolin

    echo -n "========================================================================"
    echo -n "|   The pangolin library install script has completed its execution.   |"
    echo -n "========================================================================"
}

install_pypi_packages () {
    cd "$HOME"/Monocular-VisualSLAM || exit
    check_python_venv "$(pwd)"

    python3 -m pip install -U -v    \
    opencv-python                   \
    opencv-contrib-python           \
    numpy                           \
    scikit-image                    \
    pyopengl                        \
    pyopengl-accelerate             \
    pysdl2                          \
    pysdl2-dll

    echo -n "========================================================================"
    echo -n "|   The PyPI packages install script has completed its execution.      |"
    echo -n "========================================================================"
}

install_all () {
    install_pypi_packages
    install_g2o
    install_pangolin
}

print_help () {
    echo -e "The following script arguments are supported:\n"
    echo -e "- all\tInstall all dependencies for this project to work.\n"
    echo -e "- pypi-packages\tInstall dependencies available on PyPI.\n"
    echo -e "- g2o-library\tInstall g2o library from its git repo.\n"
    echo -e "- pangolin-library\tInstall pangolin library from its git repo.\n"
}

case $1 in

    pypi-packages)
        echo -n "Installing Python dependencies available on PyPI..."
        install_pypi_packages
        ;;

    g2o-library)
        echo -n "Installing g2o python library by cloning its github repo..."
        install_g2o
        ;;

    pangolin-library)
        echo -n "Installing pangolin python library by cloning its github repo..."
        install_pangolin
        ;;
    
    all)
        echo -n "Installing all dependencies from all sources..."
        install_all
        ;;

    help | -h | -help)
        print_help
        ;;

    *)
        print_help
        ;;
esac
