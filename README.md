# PlayWithAI
This is my playpen for mucking about with AI. This should be of no use to anyone.

To get all the submodules (recommended), do a

  git clone --recurse-submodules --remote-submodules https://github.com/cpsource/PlayWithAI.git

Sometimes, repositories are not submodules and you'll see a where.from. Just cat this
and pull the module down

To create a python virtual environment, do

  python3 -m venv ~/venv

At the start of every session, you need to 'connect' to the venv. You do this by

  cd ~/
  cd Pl<tab>
  source setup-venv.sh

To setup your python3, do a

  python3 -m pip install -r requirements.txt

To backup your python3, do a

  python3 -m pip freeze > requirements.txt

Installing TensorFlow (without miniconda (barf)) can be done with the link
  https://tensorflow.org/install/pip

Misc Notes
----------

A lot of this stuff is duplicated at pythonProject having to do with learning PyCharm

As for hardware, I have an Alienware x14 gamer laptop running windows 11. The hardware has built-in Nvidia graphics card with about 2500 cuda cores, which, it turns out, will work with TensorFlow, etc.

The box also supports Ubuntu native in w11. Neet, actually, but getting it running was a bear. I made one mistake
when I purchased the box in that I didn't get enough disk space. I added a Seagate 1tb external drive to a usb port.
It's reliable but slow but works better than a Sandisk drive, which I wasted money on.

If you don't want to configure a box, run Google's Colab. It's pre-configured and comes with a Jupyter interface (I think.) Quite a sweet toy. But I prefer to set my own box up.

I created this sandbox on github to test everything.

As for getting technical help, I use bard.google.com. It's great at high-level questions, such as what is a GPU vs TPU, but really terrible with actual programming.

I use Evernote as a web clipper with a Chrome browser.

submodules
----------
Note: there are a number of config files. These are from submodules I've loaded into this top level directory.

I'm running off Ubuntu 22.04 on an Alienware x14 pc with a large monitor attached.

Notes on my pc
--------------

nvidia-smi

<date>
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.06              Driver Version: 536.40       CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060 ...    On  | 00000000:01:00.0  On |                  N/A |
| N/A   41C    P8              13W /  74W |    107MiB /  6144MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A        23      G   /Xwayland                                 N/A      |
+---------------------------------------------------------------------------------------+

uname -a

Linux AlienPC 5.15.90.1-microsoft-standard-WSL2 #1 SMP Fri Jan 27 02:56:13 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux

lsb_release -a

Distributor ID:	Ubuntu
Description:	Ubuntu 22.04.2 LTS
Release:	22.04
Codename:	jammy

nvcc --version

Command 'nvcc' not found, but can be installed with:
sudo apt install nvidia-cuda-toolkit

Oops, I didn't bother to install the nvidia cuda tool kit.

