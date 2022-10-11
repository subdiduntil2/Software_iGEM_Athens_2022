sudo apt-get -y update && sudo apt-get -y upgrade && sudo apt-get -y install build-essential python python-dev python3 python3-dev python-pip python3-pip
sudo -H pip3 install setuptools wheel numpy
sudo -H pip install setuptools wheel numpy
tar -zxvf ViennaRNA-2.5.1.tar.gz
cd ViennaRNA-2.5.1
./configure
make
sudo make install


