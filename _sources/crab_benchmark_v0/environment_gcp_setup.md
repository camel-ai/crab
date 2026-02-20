# Google cloud platform setup

## Setup and Start the VM Instance

The development image is hosted in the project `capable-vista-420022` with image name `crab-benchmark-v0-1`.

You can use [gcloud](https://cloud.google.com/sdk/docs/install) to create an instance from this image.

First install [gcloud](https://cloud.google.com/sdk/docs/install), then create an instance using the following command:

```bash
gcloud compute instances create \
crab-instance \
--zone=us-central1-a \
--machine-type=n2-standard-8 \
--image=https://www.googleapis.com/compute/v1/projects/capable-vista-420022/global/images/crab-benchmark-v0-1 \
--enable-nested-virtualization
# You can change instance name, zone, machine type as you want.
# Remember that the CPU must support nested virtualization and should have at least 32G memory.
# This setting costs around 0.4$ per hour.
```

After creating the instance, you can connect it using SSH.

User account information:

* user: `root`; password: `crab`
* user: `crab`; password: `crab`

**IMPORTANT: You must switch to user `crab` before setting up remote desktop.** Use `sudo su crab`.

## Connect the Instance through a remote desktop service

You need to connect the server to a display to set up the experiment environment because the Ubuntu virtual machine and the Android emulator require GUI operations.

There are many possible remote desktop products you can use. Here, we provide instructions for [Google Remote Desktop](https://remotedesktop.google.com/access/), which was used to run our experiment.

1. Go to [Google Remote Desktop Headless](https://remotedesktop.google.com/headless). Click **Begin** -> **Next** -> **Authorize**. On the resulting page, copy the command from the `Debian Linux` section.
2. Connect to the VM instance through SSH, paste the copied command, and run it. You will be prompted to set a six-digit PIN.
3. Go to [Google Remote Desktop Access](https://remotedesktop.google.com/access). You should see a remote device marked as online. Click it and enter the PIN. You will then see the desktop of the VM instance.