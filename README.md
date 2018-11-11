# Compiling TensorFlow from source

This side project has the goal of compiling TensorFlow from source to make the best use of my laptop's computing resources. 
The insstructions below describe my experience in installing tensorflow on my laptop which has an Intel i5-7300hq CPU, and Nvidia 1050Ti GPU and Ubuntu MATE 18.04.1 OS.


## [TensorFlow Installation Instructions](https://www.tensorflow.org/install/source)

*Note: Link used at 22.09.2018. Information provided may be different at a later date.*

### Install proper dependencies
```{shell}
$ python3 -V
Python 3.6.6
```

Because of combatibility reasons with bazel we need to use Java 8:

```
# Install java 8
$ sudo apt install openjdk-8-jdk
$ java -version
openjdk version "1.8.0_181"
OpenJDK Runtime Environment (build 1.8.0_181-8u181-b13-0ubuntu0.18.04.1-b13)
OpenJDK 64-Bit Server VM (build 25.181-b13, mixed mode)
```


### Install bazel
[Basel Website Installation Instructions](https://docs.bazel.build/versions/master/install-ubuntu.html) [Accessed at 25.09.2018]. We decide to use the bazel custom apt repository. It was the recommended method on our previous installation.
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


### Install CUDA 10

We go at [Nvidia's Website for the Ubuntu version](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu)
and download the network deb file. According to the website the installation instructions are:

```
$sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
$sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$sudo apt-get update
```
However when we proceed to install cuda we get a problem.
```
$ sudo apt-get install cuda
Reading package lists... Done
Building dependency tree       
Reading state information... Done
Some packages could not be installed. This may mean that you have
requested an impossible situation or if you are using the unstable
distribution that some required packages have not yet been created
or been moved out of Incoming.
The following information may help to resolve the situation:

The following packages have unmet dependencies.
 cuda : Depends: cuda-10-0 (>= 10.0.130) but it is not going to be installed
E: Unable to correct problems, you have held broken packages.

```

The problem is the existing graphics-drivers ppa that we have used to install the nvidia drivers. We follow this [askubuntu post](https://askubuntu.com/a/1077063/498541) explaining how to install cuda10 in Ubuntu 18.04.
First we remove the existing nvidia drivers and ppa


```
# remove graphic drivers ppa
sudo add-apt-repository -r ppa:graphics-drivers/ppa
# remove existing drivers
$ sudo apt remove nvidia-*
# remove left over packages
$ sudo apt autoremove
```
Some packages that have to do with the 390 compute drivers remained and we removed them manually.

After this we go forward with installing cuda from the already installed nvidia repository.
```
$ sudo apt install cuda
# The process requires downloading 1.5Gb of files!
# It takes some time.
```

Once we have installed cuda, we can re-enable the grapics-drivers ppa to get drivers updates as nvidias repo doesn't provide them.

Another strange thing we noticed was that the cuda drivers installed `openjdk-11-jre` despite us already having
installed `openjdk-8-jdk`.
We configure the default java as suggested by this [askubuntu answer](https://askubuntu.com/a/845300/498541)

```
# First see that we now have a new default JRE.
$ java -version
openjdk version "10.0.2" 2018-07-17
OpenJDK Runtime Environment (build 10.0.2+13-Ubuntu-1ubuntu0.18.04.2)
OpenJDK 64-Bit Server VM (build 10.0.2+13-Ubuntu-1ubuntu0.18.04.2, mixed mode)
# Check java versions installed
$ apt search openjdk | grep installed

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

default-jre/bionic,now 2:1.10-63ubuntu1~02 amd64 [installed,automatic]
default-jre-headless/bionic,now 2:1.10-63ubuntu1~02 amd64 [installed,automatic]
openjdk-11-jre/bionic-updates,bionic-security,now 10.0.2+13-1ubuntu0.18.04.2 amd64 [installed,automatic]
openjdk-11-jre-headless/bionic-updates,bionic-security,now 10.0.2+13-1ubuntu0.18.04.2 amd64 [installed,automatic]
openjdk-8-jdk/bionic-updates,bionic-security,now 8u181-b13-0ubuntu0.18.04.1 amd64 [installed]
openjdk-8-jdk-headless/bionic-updates,bionic-security,now 8u181-b13-0ubuntu0.18.04.1 amd64 [installed,automatic]
openjdk-8-jre/bionic-updates,bionic-security,now 8u181-b13-0ubuntu0.18.04.1 amd64 [installed,automatic]
openjdk-8-jre-headless/bionic-updates,bionic-security,now 8u181-b13-0ubuntu0.18.04.1 amd64 [installed,automatic]

# Configure default java version
$ sudo update-alternatives --config java
There are 2 choices for the alternative java (providing /usr/bin/java).

  Selection    Path                                            Priority   Status
------------------------------------------------------------
* 0            /usr/lib/jvm/java-11-openjdk-amd64/bin/java      1101      auto mode
  1            /usr/lib/jvm/java-11-openjdk-amd64/bin/java      1101      manual mode
  2            /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java   1081      manual mode

Press <enter> to keep the current choice[*], or type selection number: 2
update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java to provide /usr/bin/java (java) in manual mode
# Chek results.
$ java -version
openjdk version "1.8.0_181"
OpenJDK Runtime Environment (build 1.8.0_181-8u181-b13-0ubuntu0.18.04.1-b13)
OpenJDK 64-Bit Server VM (build 25.181-b13, mixed mode)
```


We also need to modify `~/.profile` to define needed environmental variables. Specifically we needd to add the following:

```
# set PATH for cuda 10.0 installation
if [ -d "/usr/local/cuda-10.0/bin/" ]; then
    export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

After that we reboot and verify our cuda 10 installation:

```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
$ nvidia-smi 
Sun Sep 30 21:20:50 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.48                 Driver Version: 410.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 105...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   49C    P0    N/A /  N/A |    173MiB /  4040MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      4013      G   /usr/lib/xorg/Xorg                           170MiB |
|    0      5328      G   /usr/lib/firefox/firefox                       1MiB |
+-----------------------------------------------------------------------------+

```


### Install CUDNN 7.3.1

We can download CUDNN 7 from [nvidia's website](https://developer.nvidia.com/rdp/cudnn-download) (requires log in).
We download the runtime, development libraries and code samples for Ubuntu 18.04.

```
# see files we downloaded
$ ll | grep cudnn
-rw-rw-r--  1 $USER $USER  149754316 Sep 29 22:01 libcudnn7_7.3.1.20-1+cuda10.0_amd64.deb
-rw-rw-r--  1 $USER $USER  137783680 Sep 29 21:55 libcudnn7-dev_7.3.1.20-1+cuda10.0_amd64.deb
-rw-rw-r--  1 $USER $USER    5231876 Sep 29 21:58 libcudnn7-doc_7.3.1.20-1+cuda10.0_amd64.deb
# Install deb files for CUDNN
$ sudo dpkg -i libcudnn7_7.3.1.20-1+cuda10.0_amd64.deb 
$ sudo dpkg -i libcudnn7-dev_7.3.1.20-1+cuda10.0_amd64.deb 
$ sudo dpkg -i libcudnn7-doc_7.3.1.20-1+cuda10.0_amd64.deb 
$ apt search libcudnn7
Sorting... Done
Full Text Search... Done
libcudnn7/now 7.3.1.20-1+cuda10.0 amd64 [installed,local]
  cuDNN runtime libraries

libcudnn7-dev/now 7.3.1.20-1+cuda10.0 amd64 [installed,local]
  cuDNN development libraries and headers

libcudnn7-doc/now 7.3.1.20-1+cuda10.0 amd64 [installed,local]
  cuDNN documents and samples


```

We then test the CUDNN installation with:

```
$ cp -r /usr/src/cudnn_samples_v7/ ~/test/
$ cd ~/test/mnistCUDNN/
$ make clean && make
$ $ ./mnistCUDNN
# ...
# Test passed!
```

Another good way to verify the installation (of a library) is from this [stackoverflow answer](https://stackoverflow.com/a/47436840/1904901):

```
function lib_installed() { /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep $1; }
function check() { lib_installed $1 && echo "$1 is installed" || echo "ERROR: $1 is NOT installed"; }
check libcudnn
	libcudnn.so.7 -> libcudnn.so.7.3.1
libcudnn is installed
```

We need to know where CUDDN7 is installed for tensorflow build configuration

```
$ whereis cudnn
cudnn: /usr/include/cudnn.h
$ ll /usr/include/ | grep cudnn
lrwxrwxrwx  1 root root     26 Sep 29 22:06 cudnn.h -> /etc/alternatives/libcudnn
$ ll /etc/alternatives/libcudnn
libcudnn        libcudnn_so     libcudnn_stlib  
$ ll /etc/alternatives/ | grep cudnn
lrwxrwxrwx   1 root root    40 Sep 29 22:06 libcudnn -> /usr/include/x86_64-linux-gnu/cudnn_v7.h
lrwxrwxrwx   1 root root    39 Sep 29 22:06 libcudnn_so -> /usr/lib/x86_64-linux-gnu/libcudnn.so.7
lrwxrwxrwx   1 root root    46 Sep 29 22:06 libcudnn_stlib -> /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a
$ dpkg-query -L libcudnn7
/.
/usr
/usr/lib
/usr/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/libcudnn.so.7.3.1
/usr/share
/usr/share/doc
/usr/share/doc/libcudnn7
/usr/share/doc/libcudnn7/changelog.Debian.gz
/usr/share/doc/libcudnn7/copyright
/usr/share/lintian
/usr/share/lintian/overrides
/usr/share/lintian/overrides/libcudnn7
/usr/lib/x86_64-linux-gnu/libcudnn.so.7
$ dpkg-query -L libcudnn7-dev 
/.
/usr
/usr/include
/usr/include/x86_64-linux-gnu
/usr/include/x86_64-linux-gnu/cudnn_v7.h
/usr/lib
/usr/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a
/usr/share
/usr/share/doc
/usr/share/doc/libcudnn7-dev
/usr/share/doc/libcudnn7-dev/changelog.Debian.gz
/usr/share/doc/libcudnn7-dev/copyright
/usr/share/lintian
/usr/share/lintian/overrides
/usr/share/lintian/overrides/libcudnn7-dev

```
We decide to use `/etc/alternatives/` for the tensorflow configuration.


### Install python3 tf build  dependencies

Below are the recommended python packages. Take care because some (e.g. numpy) may be installed through pip and not apt, in which case apt will not see them. If they are not installed in your system run:

```{shell}
$ sudo apt install python3-pip python3-dev
```

We then proceed to install tensorflow python specific dependencies:
```{shell}
$ pip3 install -U --user pip six numpy wheel mock
Collecting pip
  Downloading https://files.pythonhosted.org/packages/5f/25/e52d3f31441505a5f3af41213346e5b6c221c9e086a166f3703d2ddaf940/pip-18.0-py2.py3-none-any.whl (1.3MB)
    100% |████████████████████████████████| 1.3MB 402kB/s 
Collecting six
  Downloading https://files.pythonhosted.org/packages/67/4b/141a581104b1f6397bfa78ac9d43d8ad29a7ca43ea90a2d863fe3056e86a/six-1.11.0-py2.py3-none-any.whl
Collecting numpy
  Downloading https://files.pythonhosted.org/packages/22/02/bae88c4aaea4256d890adbf3f7cf33e59a443f9985cf91cd08a35656676a/numpy-1.15.2-cp36-cp36m-manylinux1_x86_64.whl (13.9MB)
    100% |████████████████████████████████| 13.9MB 103kB/s 
Collecting wheel
  Downloading https://files.pythonhosted.org/packages/b3/bb/42354ce8c08f66ae0cd0f4a841f40ed41d709ac9c28f292bfeb383236a4a/wheel-0.32.0-py2.py3-none-any.whl
Collecting mock
  Downloading https://files.pythonhosted.org/packages/e6/35/f187bdf23be87092bd0f1200d43d23076cee4d0dec109f195173fd3ebc79/mock-2.0.0-py2.py3-none-any.whl (56kB)
    100% |████████████████████████████████| 61kB 534kB/s 
Collecting pbr>=0.11 (from mock)
  Downloading https://files.pythonhosted.org/packages/69/1c/98cba002ed975a91a0294863d9c774cc0ebe38e05bbb65e83314550b1677/pbr-4.2.0-py2.py3-none-any.whl (100kB)
    100% |████████████████████████████████| 102kB 575kB/s 
Installing collected packages: pip, six, numpy, wheel, pbr, mock
Successfully installed mock-2.0.0 numpy-1.15.2 pbr-4.2.0 pip-18.0 six-1.11.0 wheel-0.32.0
$ pip install -U --user keras_applications==1.0.5 --no-deps
Collecting keras_applications==1.0.5
  Downloading https://files.pythonhosted.org/packages/3f/9c/6e9393ead970fd97be0cfde912697dafec5800d9191f5ba25352fa537d72/Keras_Applications-1.0.5-py2.py3-none-any.whl (44kB)
    100% |████████████████████████████████| 51kB 490kB/s 
Installing collected packages: keras-applications
Successfully installed keras-applications-1.0.5
# After pip upgrade we need to exit the shell and re-open it for pip3 command to work
# pip and pip3.6 do work fine however (doing the same thing)
$ pip3 install -U --user keras_preprocessing==1.0.3 --no-deps
Collecting keras_preprocessing==1.0.3
  Downloading https://files.pythonhosted.org/packages/b3/bd/796f986980da4d6adc77ffd8b2b11074e7b17a7b74b03789aefac5709c4b/Keras_Preprocessing-1.0.3-py2.py3-none-any.whl
Installing collected packages: keras-preprocessing
Successfully installed keras-preprocessing-1.0.3
```

### Install NCCL

When enabling CUDA tensorflow asks us to specify the nccl library location.
We go at [this nvidia webpage](https://developer.nvidia.com/nccl/nccl-download) to install it.
We download the network installer, so that it is more straightforward to get updates.

```
# See downloaded package
$ ll | grep nvidia
-rw-rw-r--  1 $USER $USER       2926 Oct  2 21:00 nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
$ sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
# Update sources
$ sudo apt update
# See new available packages
$ apt search libnccl
Sorting... Done
Full Text Search... Done
libnccl-dev/unknown 2.3.5-2+cuda10.0 amd64
  NVIDIA Collectives Communication Library (NCCL) Development Files

libnccl2/unknown 2.3.5-2+cuda10.0 amd64
  NVIDIA Collectives Communication Library (NCCL) Runtime
# Install new files
$ sudo apt install libnccl2 libnccl-dev
# See where the libraries got installed
$ dpkg-query -L libnccl-dev libnccl2
/.
/usr
/usr/include
/usr/include/nccl.h
/usr/lib
/usr/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/libnccl_static.a
/usr/share
/usr/share/doc
/usr/share/doc/libnccl-dev
/usr/share/doc/libnccl-dev/changelog.Debian.gz
/usr/share/doc/libnccl-dev/copyright
/usr/lib/x86_64-linux-gnu/libnccl.so

/.
/usr
/usr/lib
/usr/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu/libnccl.so.2.3.5
/usr/share
/usr/share/doc
/usr/share/doc/libnccl2
/usr/share/doc/libnccl2/changelog.Debian.gz
/usr/share/doc/libnccl2/copyright
/usr/lib/x86_64-linux-gnu/libnccl.so.2
```

The packages are being installed in locations that tensorflow can't find them:

```
Invalid path to NCCL 2 toolkit, /usr/include/lib/libnccl.so.2 or /usr/include/include/nccl.h not found.
```
We decide to create symbolic links from within the cuda installation to where they are:
```
$ cd /usr/local/cuda/
$ sudo mkdir lib
$ cd lib
$ sudo ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 libnccl.so.2
$ cd ../include
$ sudo ln -s /usr/include/nccl.h nccl.h
```
This way we can point tensorflow to the default cuda directory for nccl.

### Go at the TensorFlow source code repository
Clone the tensorflow repository locally. 
```{shell}
$ cd ~/Repositories/tensorflow/
# go to r1.11 branch
$ git checkout r1.11
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
WARNING: An illegal reflective access operation has occurred
WARNING: Illegal reflective access by com.google.protobuf.UnsafeUtil (file:/home/$USER/.cache/bazel/_bazel_$USER/install/792a28b07894763eaa2bd870f8776b23/_embedded_binaries/A-server.jar) to field java.lang.String.value
WARNING: Please consider reporting this to the maintainers of com.google.protobuf.UnsafeUtil
WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
WARNING: All illegal access operations will be denied in a future release
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.17.2 installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.6/dist-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: y
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon AWS Platform support? [Y/n]: n
No Amazon AWS Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with nGraph support? [y/N]: n
No nGraph support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 10.0


Please specify the location where CUDA 10.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.3


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /etc/alternatives/


Do you wish to build TensorFlow with TensorRT support? [y/N]: n
No TensorRT support will be enabled for TensorFlow.

Please specify the NCCL version you want to use. If NCCL 2.2 is not installed, then you can use version 1.3 that can be fetched automatically but it may have worse performance with multiple GPUs. [Default is 2.2]: 2.3


Please specify the location where NCCL 2 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 6.1]: 6.1


Do you want to use clang as CUDA compiler? [y/N]: n
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
Configuration finished
```

### Configuration options
Some helpful links to understand the configuration options:
- [Intel Math Kernel Library (MKL)](https://en.wikipedia.org/wiki/Math_Kernel_Library). Enabled it even though I don't think it 'll contribute much because of the dedicated GPU.
- [VERBS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/verbs): This appears to be a [Remote direct memory access](https://en.wikipedia.org/wiki/Remote_direct_memory_access) feature. Since I only plan to use tf on my laptop this was disabled.
- According to [wikipedia, MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) is used in parallel computing set ups and I don't think this has a use case for my laptop hence I didn't enable it.
- According to Nvidia [GDR](https://developer.nvidia.com/gpudirect) stands for GPU Direct. Using GPU direct various devices can read and write CUDA host and device memory. (...)

### Run source tree tests

If we run those tests before configuring the instllation, as suggested in the [tensorflow instructions](https://www.tensorflow.org/install/source)
we get an error because numpy was not found. But if we run them after the configuration has finished they run:

```
$ bazel test -c opt -- //tensorflow/... -//tensorflow/compiler/... -//tensorflow/contrib/lite/...
# We get a lot of warnings during the tests compilaiton that we don't include here.
```



### Build the pip package with GPU support:
Moving forward we build tensorflow:
```{shell}
$ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package  > logCompile.txt 2>&1
```
The build process gives a huge number of wanring messages captured in [this file](./logCompile.txt).

Subsequently we build the pip/wheel binary:

```{shell}
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./
Wed 3 Oct 00:00:58 BST 2018 : === Preparing sources in dir: /tmp/tmp.yPuC7bp2cE
/xtras/$USER/Repositories/tensorflow /xtras/$USER/Repositories/tensorflow
/xtras/$USER/Repositories/tensorflow
Wed 3 Oct 00:01:06 BST 2018 : === Building wheel
warning: no files found matching '*.pd' under directory '*'
warning: no files found matching '*.dll' under directory '*'
warning: no files found matching '*.lib' under directory '*'
warning: no files found matching '*.h' under directory 'tensorflow/include/tensorflow'
warning: no files found matching '*' under directory 'tensorflow/include/Eigen'
warning: no files found matching '*.h' under directory 'tensorflow/include/google'
warning: no files found matching '*' under directory 'tensorflow/include/third_party'
warning: no files found matching '*' under directory 'tensorflow/include/unsupported'
Wed 3 Oct 00:01:24 BST 2018 : === Output wheel file is in: /xtras/$USER/Repositories/tensorflow/

```
- Comprehension Question: Why did we need to do it in two steps? <br> _What does the first step do if not create the binary of the compiled program? (And if so where are the output files placed?)_

Finally let's install our new package! _(Installing packages globally as root is NOT a good practive.)_

```{shell}
$ pip3 install -U --user ./tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl 
Processing ./tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl
Requirement already satisfied, skipping upgrade: setuptools<=39.1.0 in /usr/lib/python3/dist-packages (from tensorflow==1.11.0) (39.0.1)
Collecting astor>=0.6.0 (from tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/35/6b/11530768cac581a12952a2aad00e1526b89d242d0b9f59534ef6e6a1752f/astor-0.7.1-py2.py3-none-any.whl
Collecting absl-py>=0.1.6 (from tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/16/db/cce5331638138c178dd1d5fb69f3f55eb3787a12efd9177177ae203e847f/absl-py-0.5.0.tar.gz (90kB)
    100% |████████████████████████████████| 92kB 115kB/s 
Collecting tensorboard<1.12.0,>=1.11.0 (from tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/9b/2f/4d788919b1feef04624d63ed6ea45a49d1d1c834199ec50716edb5d310f4/tensorboard-1.11.0-py3-none-any.whl (3.0MB)
    100% |████████████████████████████████| 3.0MB 557kB/s 
Collecting protobuf>=3.6.0 (from tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/c2/f9/28787754923612ca9bfdffc588daa05580ed70698add063a5629d1a4209d/protobuf-3.6.1-cp36-cp36m-manylinux1_x86_64.whl (1.1MB)
    100% |████████████████████████████████| 1.1MB 558kB/s 
Requirement already satisfied, skipping upgrade: keras-applications>=1.0.5 in /home/$USER/.local/lib/python3.6/site-packages (from tensorflow==1.11.0) (1.0.5)
Requirement already satisfied, skipping upgrade: numpy>=1.13.3 in /home/$USER/.local/lib/python3.6/site-packages (from tensorflow==1.11.0) (1.15.2)
Collecting grpcio>=1.8.6 (from tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/a7/9c/523fec4e50cd4de5effeade9fab6c1da32e7e1d72372e8e514274ffb6509/grpcio-1.15.0-cp36-cp36m-manylinux1_x86_64.whl (9.5MB)
    100% |████████████████████████████████| 9.5MB 514kB/s 
Requirement already satisfied, skipping upgrade: keras-preprocessing>=1.0.3 in /home/$USER/.local/lib/python3.6/site-packages (from tensorflow==1.11.0) (1.0.3)
Requirement already satisfied, skipping upgrade: six>=1.10.0 in /home/$USER/.local/lib/python3.6/site-packages (from tensorflow==1.11.0) (1.11.0)
Requirement already satisfied, skipping upgrade: wheel>=0.26 in /home/$USER/.local/lib/python3.6/site-packages (from tensorflow==1.11.0) (0.32.0)
Collecting gast>=0.2.0 (from tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/5c/78/ff794fcae2ce8aa6323e789d1f8b3b7765f601e7702726f430e814822b96/gast-0.2.0.tar.gz
Collecting termcolor>=1.1.0 (from tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz
Collecting markdown>=2.6.8 (from tensorboard<1.12.0,>=1.11.0->tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/7a/6b/5600647404ba15545ec37d2f7f58844d690baf2f81f3a60b862e48f29287/Markdown-3.0.1-py2.py3-none-any.whl (89kB)
    100% |████████████████████████████████| 92kB 674kB/s 
Collecting werkzeug>=0.11.10 (from tensorboard<1.12.0,>=1.11.0->tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/20/c4/12e3e56473e52375aa29c4764e70d1b8f3efa6682bef8d0aae04fe335243/Werkzeug-0.14.1-py2.py3-none-any.whl (322kB)
    100% |████████████████████████████████| 327kB 578kB/s 
Collecting h5py (from keras-applications>=1.0.5->tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/8e/cb/726134109e7bd71d98d1fcc717ffe051767aac42ede0e7326fd1787e5d64/h5py-2.8.0-cp36-cp36m-manylinux1_x86_64.whl (2.8MB)
    100% |████████████████████████████████| 2.8MB 567kB/s 
Collecting keras>=2.1.6 (from keras-applications>=1.0.5->tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/06/ea/ad52366ce566f7b54d36834f98868f743ea81a416b3665459a9728287728/Keras-2.2.3-py2.py3-none-any.whl (312kB)
    100% |████████████████████████████████| 317kB 648kB/s 
Collecting scipy>=0.14 (from keras-preprocessing>=1.0.3->tensorflow==1.11.0)
  Downloading https://files.pythonhosted.org/packages/a8/0b/f163da98d3a01b3e0ef1cab8dd2123c34aee2bafbb1c5bffa354cc8a1730/scipy-1.1.0-cp36-cp36m-manylinux1_x86_64.whl (31.2MB)
    100% |████████████████████████████████| 31.2MB 482kB/s 
Requirement already satisfied, skipping upgrade: pyyaml in /usr/lib/python3/dist-packages (from keras>=2.1.6->keras-applications>=1.0.5->tensorflow==1.11.0) (3.12)
Building wheels for collected packages: absl-py, gast, termcolor
  Running setup.py bdist_wheel for absl-py ... done
  Stored in directory: /home/$USER/.cache/pip/wheels/3c/33/ae/db8cd618e62f87594c13a5483f96e618044f9b01596efd013f
  Running setup.py bdist_wheel for gast ... done
  Stored in directory: /home/$USER/.cache/pip/wheels/9a/1f/0e/3cde98113222b853e98fc0a8e9924480a3e25f1b4008cedb4f
  Running setup.py bdist_wheel for termcolor ... done
  Stored in directory: /home/$USER/.cache/pip/wheels/7c/06/54/bc84598ba1daf8f970247f550b175aaaee85f68b4b0c5ab2c6
Successfully built absl-py gast termcolor
keras 2.2.3 has requirement keras-applications>=1.0.6, but you'll have keras-applications 1.0.5 which is incompatible.
keras 2.2.3 has requirement keras-preprocessing>=1.0.5, but you'll have keras-preprocessing 1.0.3 which is incompatible.
Installing collected packages: astor, absl-py, protobuf, markdown, werkzeug, grpcio, tensorboard, gast, termcolor, tensorflow, h5py, scipy, keras
Successfully installed absl-py-0.5.0 astor-0.7.1 gast-0.2.0 grpcio-1.15.0 h5py-2.8.0 keras-2.2.3 markdown-3.0.1 protobuf-3.6.1 scipy-1.1.0 tensorboard-1.11.0 tensorflow-1.11.0 termcolor-1.1.0 werkzeug-0.14.1

```

### Testing the new installation
We test the new installation on the terminal:
```{shell}
# We shouldn't import tensorflow from source directory!
$ cd ~/test/
$ python3
Python 3.6.6 (default, Sep 12 2018, 18:26:19) 
[GCC 8.0.1 20180414 (experimental) [trunk revision 259383]] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version__)
1.11.0-iolaum1
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
2018-10-03 00:07:48.599927: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-10-03 00:07:48.600337: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1411] Found device 0 with properties: 
name: GeForce GTX 1050 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.62
pciBusID: 0000:01:00.0
totalMemory: 3.95GiB freeMemory: 3.68GiB
2018-10-03 00:07:48.600351: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1490] Adding visible gpu devices: 0
2018-10-03 00:07:48.811302: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-03 00:07:48.811328: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977]      0 
2018-10-03 00:07:48.811351: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990] 0:   N 
2018-10-03 00:07:48.811494: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1103] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3409 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
>>> print(sess.run(hello))
b'Hello, TensorFlow!'
>>> exit()


```

## Miscalleneous Notes.

- When compiling from source the tensorflow package is called ```tensorflow``` (for pip) but when installing the precompiled binary it can be called either ```tensorflow``` or ```tensorflow-gpu```. Hence if you are replacing the gpu version with a compiled one be sure to manually uninstall it first to avoid ending up with both installed in your system.

- According to [this stack overflow answer](https://stackoverflow.com/a/35963479/1904901)
starting python and then importing tensorflow within the tensorflow source code repository may cause problems !

