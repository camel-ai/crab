## Setup and Start the VM Instance

TODO

## Connect the Instance through a remote desktop service

You need to connect the server to a display to set up the experiment environment because the Ubuntu virtual machine and the Android emulator require GUI operations.

There are many possible remote desktop products you can use. Here, we provide instructions for [Google Remote Desktop](https://remotedesktop.google.com/access/), which was used to run our experiment.

1. Go to [Google Remote Desktop Headless](https://remotedesktop.google.com/headless). Click **Begin** -> **Next** -> **Authorize**. On the resulting page, copy the command from the `Debian Linux` section.
2. Connect to the VM instance through SSH, paste the copied command, and run it. You will be prompted to set a six-digit PIN.
3. Go to [Google Remote Desktop Access](https://remotedesktop.google.com/access). You should see a remote device marked as online. Click it and enter the PIN. You will then see the desktop of the VM instance.

## 