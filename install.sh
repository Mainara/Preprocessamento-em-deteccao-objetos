#!/bin/bash

sudo apt-get update
sudo apt-get -y install python-pip
sudo pip install virtualenv
virtualenv -p python .envPC && . .envPC/bin/activate

pip install -r requirements.txt

