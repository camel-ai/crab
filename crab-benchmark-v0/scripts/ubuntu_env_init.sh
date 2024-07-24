#!/bin/bash

# Disable screen autolock
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.desktop.session idle-delay 0

# Disable automatic updates
sudo bash -c 'cat <<EOF > /etc/apt/apt.conf.d/20auto-upgrades
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Unattended-Upgrade "0";
EOF'

# Allow sudo without password for the current user
CURRENT_USER=$(whoami)
sudo bash -c "echo \"$CURRENT_USER ALL=(ALL) NOPASSWD: ALL\" | tee /etc/sudoers.d/$CURRENT_USER"

# Install required packages
sudo apt update
sudo apt install -y openssh-server git vim python3-pip xdotool python3-tk python3.10-venv

# Install pipx
python3 -m pip install pipx
python3 -m pipx ensurepath

# Modify .bashrc to alias python to python3 for the current user
echo 'alias python=python3' >> /home/$CURRENT_USER/.bashrc

# Reload .bashrc for the current user
source /home/$CURRENT_USER/.bashrc

# Install poetry using pipx
pipx install poetry

# Pull CRAB repo
if [ ! -d "/home/$CURRENT_USER/crab" ]; then
    git clone https://github.com/camel-ai/crab.git /home/$CURRENT_USER/crab/
fi

# Create poetry environment
cd /home/$CURRENT_USER/crab
poetry install -E server

# Change to X11 from Wayland
sudo sed -i 's/#WaylandEnable=false/WaylandEnable=false/g' /etc/gdm3/custom.conf
touch /home/$CURRENT_USER/.Xauthority

# Create the crab.service file with dynamic user and group
sudo bash -c "cat <<EOF > /etc/systemd/system/crab.service
[Unit]
Description=My Python Script Service
After=network.target

[Service]
WorkingDirectory=/home/$CURRENT_USER/crab/
ExecStart=/home/$CURRENT_USER/.local/bin/poetry run python -m crab.server.main --HOST 0.0.0.0
Restart=always
User=$CURRENT_USER
Group=$CURRENT_USER

[Install]
WantedBy=multi-user.target
EOF"

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable and start the crab service
sudo systemctl enable crab.service

# Reboot the system to apply changes for X11
echo "System will reboot in 10 seconds to apply changes..."
sleep 10
sudo reboot