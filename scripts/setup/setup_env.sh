# This script is used to setup assisi-gym environment on Ubuntu 20.04
#  install pyenv
curl https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL"
# install and activate python in pyenv
pyenv install 3.7.10
eval "$(pyenv init -)"
pyenv global 3.7.10


# clone assisi-gym and install
git clone https://github.com/longnguyen2706/assistive-gym.git
pip3 install --upgrade pip
pip3 install git+https://github.com/Healthcare-Robotics/assistive-gym.git

# install additional dependencies
sudo apt-get install python-tk python3-tk tk-dev
sudo apt-get install build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev

# downgrade protobuf 
pip3 install protobuf==3.20.*
pip3 install pytorch3
pip3 install opencv-python
pip3 install matplotlib
pip3 install chumpy
pip3 install open3d # for trimesh decimation
pip3 install pytorch3d
pip3 install cma
pip3 install cmaes
pip3 install kinpy
pip3 install ergonomics-metrics # can remove if we dont need reba score
pip3 install keyboard
pip3 install deprecation 

pip3 install pybullet==3.2.5 # upgrade to 3.2.5 (latest version) so that we can have collision detection without running physic steps