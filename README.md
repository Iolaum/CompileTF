# Compiling TensorFlow from source.

This side project has the goal of compiling TensorFlow from source to make the best use of my laptop's computing resources. The insstructions below describe my experience in installing tensorflow on my laptop which has an Intel i5-7300hq CPU, and Nvidia 1050Ti GPU and Ubuntu 16.04.3 OS.



## [TensorFlow Installation Instructions](https://www.tensorflow.org/install/install_sources)

*Note: Link used at 16.06.2017. Information provided may be different at a later date.*

### Perform preparing environment for Linux instructions. (Python and Java dependencies)
```{shell}
$ python3 -V
Python 3.5.2
$ java -version
openjdk version "1.8.0_131"
OpenJDK Runtime Environment (build 1.8.0_131-8u131-b11-0ubuntu1.16.04.2-b11)
OpenJDK 64-Bit Server VM (build 25.131-b11, mixed mode)
```

### Install bazel
[Basel Website Installation Instructions](https://bazel.build/versions/master/docs/install.html) [Accessed at 4.07.2017]. We decide to use the bazel custom apt repository (which is the recommended method).

```{shell}
# Add Bazel distribution URI as a package source (one time setup)
$ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
$ curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install bazel
# Note: This time java's ibm was not installed !
$ java -version
openjdk version "1.8.0_131"
OpenJDK Runtime Environment (build 1.8.0_131-8u131-b11-0ubuntu1.16.04.2-b11)
OpenJDK 64-Bit Server VM (build 25.131-b11, mixed mode)

```

### Install python3.5 tf build  dependencies

Below are the recommended python packages. Take care because some (e.g. numpy) may be installed through pip and not apt, in which case apt will not see them. If they are not installed in your system run:

```{shell}
$ sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel
```

### Install CUDA 8

We go through the [nvidia instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/) to install CUDA 8.

**Note on post installation instructions:**<br>
We need to set two permanent environmental variables. According to [ubuntu's documentation](https://help.ubuntu.com/community/EnvironmentVariables#Persistent_environment_variables)
we chose the following way to add the relevant commands at the ```.profile``` file. However because of a bug that causes a login loop in ubuntu we add them in the ```.bash_aliases``` file.

```
$ echo -e "\n\n# Persistent Environmental variables needed for CUDA + TensorFlow" >> .bash_aliases
$ echo -e "# Set LD_LIBRARY_PATH here to avoid login loop when setting it in .profile" >> .bash_aliases
$ echo -e "export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}" >> .bash_aliases
$ echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/lib/nvidia-384${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> .bash_aliases
```

Note the hard-coded nvidia drivers directory. If you change drivers version (e.g. from 375 to 384) make sure to update LD\_LIBRARY\_PATH.

To verify our installation we can run the following command:

```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```


### Install CUDNN 6

You can download CUDNN 6 from [nvidia's website](https://developer.nvidia.com/cudnn) (requires log in).

Then we go to where we downloaded the file and:

```
# Go to Downloads directory
$ cd ~/Downloads
# unzip file
$ tar -zxvf cudnn-8.0-linux-x64-v6.0.tgz 
cuda/include/cudnn.h
cuda/lib64/libcudnn.so
cuda/lib64/libcudnn.so.6
cuda/lib64/libcudnn.so.6.0.21
cuda/lib64/libcudnn_static.a
```
We need to replace the files from the previous version with those files. Hence we go to delete the previous ones first.

```
$ cd /usr/local/cuda-8.0/
$ ls -la include/*cudnn*
-r--r--r-- 1 root root 99658 Mar  2 23:34 include/cudnn.h
$ ls -la lib64/*cudnn*
lrwxrwxrwx 1 root root       13 Mar  2 23:35 lib64/libcudnn.so -> libcudnn.so.5
lrwxrwxrwx 1 root root       18 Mar  2 23:35 lib64/libcudnn.so.5 -> libcudnn.so.5.1.10
-rwxr-xr-x 1 root root 84163560 Mar  2 23:35 lib64/libcudnn.so.5.1.10
-rw-r--r-- 1 root root 70364814 Mar  2 23:35 lib64/libcudnn_static.a
```
Those are the files we want to delete and replace with the new ones, therefore:
```
# sudo is needed because those files are outside the user's home directory
$ sudo rm include/*cudnn*
$ sudo rm lib64/*cudnn*
```

No we proceed to copy the new CUDNN 6 files:

```
# sudo is needed because we are copying files outside of the user's home directory
# -P is needed to preserve symbolic links
$ sudo cp -P ~/Downloads/cuda/include/cudnn.h /usr/local/cuda-8.0/include/
$ sudo cp -P ~/Downloads/cuda/lib64/*cudnn* /usr/local/cuda-8.0/lib64/
```

### Install libcupti-dev 

In case it's not already installed:

```{shell}
$ sudo apt-get install libcupti-dev
```

### Go at the TensorFlow source code repository
Clone the tensorflow repository locally. 
```{shell}
$ cd ~/Repositories/tensorflow/
# go to r1.4 branch
$ git checkout r1.4
```
In the case of a previously compiled library we need to run
```{shell}
$ bazel clean
```
to clean bazel temporary configuration files. (We also need to update the repository if needed.)

### Configuring the compilation:
After checking out on the current release branch we start the configuration script and enter the choices described below:
```{shell}
$ ./configure 
WARNING: Running Bazel server needs to be killed, because the startup options are different.
You have bazel 0.7.0 installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib/python3.5/dist-packages
  /usr/lib/python3/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python3.5/dist-packages]
/usr/local/lib/python3.5/dist-packages
Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: y
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL support? [y/N]: n
No OpenCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 8.0]: 


Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 6.0]: 


Please specify the location where cuDNN 6 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,5.2]6.1


Do you want to use clang as CUDA compiler? [y/N]: n
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Add "--config=mkl" to your bazel command to build with MKL support.
Please note that MKL on MacOS or windows is still not supported.
If you would like to use a local MKL instead of downloading, please set the environment variable "TF_MKL_ROOT" every time before build.
Configuration finished

```

### Configuration options
Some helpful links to understand the configuration options:
- [Intel Math Kernel Library (MKL)](https://en.wikipedia.org/wiki/Math_Kernel_Library). Enabled it even though I don't think it 'll contribute much because of the dedicated GPU.
- [VERBS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/verbs): This appears to be a [Remote direct memory access](https://en.wikipedia.org/wiki/Remote_direct_memory_access) feature. Since I only plan to use tf on my laptop this was disabled.
- According to [wikipedia, MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) is used in parallel computing set ups and I don't think this has a use case for my laptop hence I didn't enable it.
- According to Nvidia [GDR](https://developer.nvidia.com/gpudirect) stands for GPU Direct. Using GPU direct various devices can read and write CUDA host and device memory. (...)

### Build the pip package with GPU support:
Moving forward we build tensorflow:
```{shell}
$ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package  > logCompile.txt 2>&1
```
The build process gives a huge number of wanring messages captured in [this file](./logCompile.txt).

Subsequently we build the pip/wheel binary:

```{shell}
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package ./pip_packages/
Fri 3 Nov 18:47:02 GMT 2017 : === Using tmpdir: /tmp/tmp.QTbHIFLHK9
~/Repositories/tensorflow/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles ~/Repositories/tensorflow
~/Repositories/tensorflow
/tmp/tmp.QTbHIFLHK9 ~/Repositories/tensorflow
Fri 3 Nov 18:47:03 GMT 2017 : === Building wheel
warning: no files found matching '*.dll' under directory '*'
warning: no files found matching '*.lib' under directory '*'
warning: no files found matching '*.h' under directory 'tensorflow/include/tensorflow'
warning: no files found matching '*' under directory 'tensorflow/include/Eigen'
warning: no files found matching '*' under directory 'tensorflow/include/external'
warning: no files found matching '*.h' under directory 'tensorflow/include/google'
warning: no files found matching '*' under directory 'tensorflow/include/third_party'
warning: no files found matching '*' under directory 'tensorflow/include/unsupported'
~/Repositories/tensorflow
Fri 3 Nov 18:47:15 GMT 2017 : === Output wheel file is in: /home/$USER/Repositories/tensorflow/pip_packages/

```
- Comprehension Question: Why did we need to do it in two steps? <br> _What does the first step do if not create the binary of the compiled program? (And if so where are the output files placed?)_

Finally let's install our new package! _(Installing packages globally as root is NOT a good practive.)_

```{shell}
$ sudo -H pip3 install ./pip_packages/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl 
[sudo] password for $USER: 
Processing ./pip_packages/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl
Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.5/dist-packages (from tensorflow==1.4.0)
Requirement already satisfied: protobuf>=3.3.0 in /usr/local/lib/python3.5/dist-packages (from tensorflow==1.4.0)
Collecting enum34>=1.1.6 (from tensorflow==1.4.0)
  Downloading enum34-1.1.6-py3-none-any.whl
Collecting tensorflow-tensorboard<0.5.0,>=0.4.0rc1 (from tensorflow==1.4.0)
  Downloading tensorflow_tensorboard-0.4.0rc2-py3-none-any.whl (1.7MB)
    100% |████████████████████████████████| 1.7MB 605kB/s 
Requirement already satisfied: numpy>=1.12.1 in /usr/local/lib/python3.5/dist-packages (from tensorflow==1.4.0)
Requirement already satisfied: six>=1.10.0 in /usr/lib/python3/dist-packages (from tensorflow==1.4.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.5/dist-packages (from protobuf>=3.3.0->tensorflow==1.4.0)
Collecting html5lib==0.9999999 (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow==1.4.0)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.5/dist-packages (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow==1.4.0)
Collecting bleach==1.5.0 (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow==1.4.0)
  Using cached bleach-1.5.0-py2.py3-none-any.whl
Requirement already satisfied: werkzeug>=0.11.10 in /usr/local/lib/python3.5/dist-packages (from tensorflow-tensorboard<0.5.0,>=0.4.0rc1->tensorflow==1.4.0)
Installing collected packages: enum34, html5lib, bleach, tensorflow-tensorboard, tensorflow
  Found existing installation: html5lib 0.999999999
    Uninstalling html5lib-0.999999999:
      Successfully uninstalled html5lib-0.999999999
  Found existing installation: bleach 2.0.0
    Uninstalling bleach-2.0.0:
      Successfully uninstalled bleach-2.0.0
  Found existing installation: tensorflow-tensorboard 0.1.6
    Uninstalling tensorflow-tensorboard-0.1.6:
      Successfully uninstalled tensorflow-tensorboard-0.1.6
  Found existing installation: tensorflow 1.3.0
    Uninstalling tensorflow-1.3.0:
      Successfully uninstalled tensorflow-1.3.0
Successfully installed bleach-1.5.0 enum34-1.1.6 html5lib-0.9999999 tensorflow-1.4.0 tensorflow-tensorboard-0.4.0rc2

```

### Testing the new installation
We test the new installation on the terminal:
```{shell}
$ cd ~
$ python3
Python 3.5.2 (default, Sep 14 2017, 22:51:06) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version__)
1.4.0
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
2017-11-03 20:38:16.717173: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-11-03 20:38:16.717422: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: GeForce GTX 1050 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.62
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 3.62GiB
2017-11-03 20:38:16.717433: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
>>> print(sess.run(hello))
b'Hello, TensorFlow!'
>>> exit()


```

## Miscalleneous Notes.

- When compiling from source the tensorflow package is called ```tensorflow``` (for pip) but when installing the precompiled binary it can be called either ```tensorflow``` or ```tensorflow-gpu```. Hence if you are replacing the gpu version with a compiled one be sure to manually uninstall it first to avoid ending up with both installed in your system.

- According to [this stack overflow answer](https://stackoverflow.com/a/35963479/1904901)
starting python and then importing tensorflow within the tensorflow source code repository may cause problems !

- After the successful compilation of version 1.4 we had the following error:

```
$ cd ~
$ python3
Python 3.5.2 (default, Sep 14 2017, 22:51:06) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/__init__.py", line 24, in <module>
    from tensorflow.python import *
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/__init__.py", line 83, in <module>
    from tensorflow.python.estimator import estimator_lib as estimator
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/estimator/estimator_lib.py", line 35, in <module>
    from tensorflow.python.estimator.inputs import inputs
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/estimator/inputs/inputs.py", line 22, in <module>
    from tensorflow.python.estimator.inputs.numpy_io import numpy_input_fn
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/estimator/inputs/numpy_io.py", line 22, in <module>
    from tensorflow.python.estimator.inputs.queues import feeding_functions
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/estimator/inputs/queues/feeding_functions.py", line 40, in <module>
    import pandas as pd
  File "/usr/local/lib/python3.5/dist-packages/pandas/__init__.py", line 60, in <module>
    import pandas.testing
  File "/usr/local/lib/python3.5/dist-packages/pandas/testing.py", line 7, in <module>
    from pandas.util.testing import (
  File "/usr/local/lib/python3.5/dist-packages/pandas/util/testing.py", line 55, in <module>
    slow = pytest.mark.slow
AttributeError: module 'pytest' has no attribute 'mark'
>>> exit()
```

This happened because of pandas:

```
$ python3
Python 3.5.2 (default, Sep 14 2017, 22:51:06) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> import pandas as pd
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.5/dist-packages/pandas/__init__.py", line 60, in <module>
    import pandas.testing
  File "/usr/local/lib/python3.5/dist-packages/pandas/testing.py", line 7, in <module>
    from pandas.util.testing import (
  File "/usr/local/lib/python3.5/dist-packages/pandas/util/testing.py", line 55, in <module>
    slow = pytest.mark.slow
AttributeError: module 'pytest' has no attribute 'mark'
>>> exit()
```

Upgrading to a newer pandas version fixed the problem.
