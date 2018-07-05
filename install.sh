# remove dir env
if [ -d "$env_dir" ]
then
    rm -r .env/
fi

sudo apt-get update
sudo apt-get -y install python-pip
sudo pip install virtualenv
virtualenv -p python .env && . .env/bin/activate

.env/bin/pip install -r requirements.txt