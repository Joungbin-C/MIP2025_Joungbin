# Isaac Sim Environments Setup

### 1. Isaac Sim Requirements

| Element | Minimum Spec                                     | Good                                            | Ideal                                                        |
| ------- | ------------------------------------------------ | ----------------------------------------------- | ------------------------------------------------------------ |
| OS      | Ubuntu 22.04/24.04, <br />Windows 10/11          | Ubuntu 22.04/24.04, <br />Windows 10/11         | Ubuntu 22.04/24.04, <br />Windows 10/11                      |
| CPU     | Intel Core i7 (7th Generation)<br /> AMD Ryzen 5 | Intel Core i7 (9th Generation)<br />AMD Ryzen 7 | Intel Core i9, X-series or higher<br />AMD Ryzen9, Threadripper or higher |
| Cores   | 4                                                | 8                                               | 16                                                           |
| Ram     | 32GB                                             | 64GB                                            | 64GB                                                         |
| Storage | 50GB SSD                                         | 500GB SSD                                       | 1TB NVMe SSD                                                 |
| GPU     | 16GB                                             | 16GB                                            | 48GB                                                         |
| VRAM    | 16GB                                             | 16GB                                            | 48GB                                                         |
| Driver  | Linux: 580.65.06<br />Windows: 580.88            | Linux: 580.65.06<br />Windows: 580.88           | Linux: 580.65.06<br />Windows: 580.88                        |

You can run the [Isaac Sim Compatibility Checker][https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html#isaac-sim-compatibility-checker] to check if your machine meets the system requirements and compatibility.



### 2. Isaac Sim Setup

#### 2.1 Download

[download link](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html)

Go into the download link and download the lastest release that meets the os system with your machine.



#### 2.2 Isaac Sim Lauch

1. 1. Download the [Latest Release](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release) of **Isaac Sim** for your platform to the `Downloads` folder.

   2. Create a folder named `isaacsim` at `c:/` or at the root of your Linux environment.

   3. Unzip the package to that folder.

   4. Navigate to that folder.

   5. To create a symlink to the **extension_examples** for the tutorials, run the `post_install` script. The script can be run at this stage or after installation.

      > - On Linux, run `./post_install.sh`.
      > - On Windows, double click `post_install.bat`.

2. 6. Use one of the following methods to run **Isaac Sim**:
      - On Linux, run `./isaac-sim.sh`.
      - On Windows, run `isaac-sim.bat`.



#### 2.3 Preparation

Make sure that you have python 3.11, Conda, PyCharm(or VScode) in your machine. If you don't have them, follow the link and install them:

-  [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- [Conda for window](https://docs.conda.io/projects/conda/en/stable/user-guide/install/windows.html) or [Conda for Linux](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)
- [PyCharm](https://www.jetbrains.com/ko-kr/pycharm/download/?section=windows) or [VSCode](https://code.visualstudio.com/download)

To operate GPU, cuda and cuDNN must be installed.

- CUDA Installation

  - On command Prompt, run `nvdia-smi`
  - CUDA Version(top-right) → Maximum compatible CUDA version

  ![cuda](https://github.com/Joungbin-C/MIP2025_Joungbin/blob/main/img/cuda.png)

  - [Installation link](https://developer.nvidia.com/cuda-toolkit-archive)

- cuDNN Installation

  - cuDNN version: v8.6.0 for CUDA 13.x

  - [Installation link](https://developer.nvidia.com/rdp/cudnn-archive)

  - After installation
     → Copy the **bin**, **include**, and **lib** folders from the **cuDNN** directory
     → Paste them directly into the **CUDA installation directory**

  - Author’s CUDA installation path:

    ```
    C drive → Program Files → NVIDIA GPU Computing Toolkit → CUDA → v13.x
    ```

  - If the process is completed correctly:

    - Open Command Prompt

    - Run `nvcc -V`

    - Output should appear as shown below

      ![cudnn](https://github.com/Joungbin-C/MIP2025_Joungbin/blob/main/img/cudnn.png)



### Isaac Sim Python Environment Installation

1. Create and activate the virtual environment 

```
conda create -n env_isaacsim python=3.11
conda activate env_isaacsim
```

2. Install *Isaac Sim - Python packages*:

```
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
```

3. After it, run the python sample scripts

```
python path/to/script.py
```

4. Setup the interpreter as `env_isaacsim`



### Library Installation

1. Activate the virtual enviroment

```
conda activate env_isaacsim
```

2. install libraries

```
# Mathmatics, Vision
pip install numpy==1.26.4
pip install scipy==1.16.3
pip install opencv-python==4.11.0.86
pip install open3d==0.19.0

# Deep learning, YOLO
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.3.235

# Plotting
pip install matplotlib==3.10.3
pip install pillow==11.3.0
```

3. Check the installation

```
python -c "import numpy, scipy, cv2, open3d, torch, ultralytics"
```



# PiPER Robot Arm Operation

Read the `PiPER Start Manual.pdf` and install it in Linux environments

```
MIP2025_Joungbin → Software → PickandPlace_piper → PiPER Start Manual.pdf
```

There is a file `Pallet_PickandPlace_jb` in `piper_sdk` folder. Copy and paste to your `piper_sdk` folder and run `detector_with_robot.py`

Before running it, make sure ZED sdk and libraries are all installed

1. PiPER Python SDK Installation : start manual page 21~31 or [github](https://github.com/agilexrobotics/piper_sdk/tree/master?tab=readme-ov-file)

2. [ZED sdk installation](https://www.stereolabs.com/en-kr/developers/release) : match your cuda version with zed sdk version

3. [ZED python API installation](https://www.stereolabs.com/docs/development/python/install)

   1. 확인 코드 `python -c "import pyzed.sl as sl"`

4. Python Libraries

   ```
   pip install numpy==1.26.4
   pip install opencv-python==4.11.0.86
   pip install scipy==1.16.3
   pip install open3d==0.19.0
   
   pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
     --index-url https://download.pytorch.org/whl/cu128
   pip install ultralytics==8.3.235
   
   pip install pyzed
   pip install argparse
   ```

5. Two command prompt

   1. ```
      bash can_activate.sh can0 1000000 
      ```

   2. ```
      cd path/to/detector_with_robot.py
      python detector_with_robot.py
      ```


      



