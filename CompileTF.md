# Compiling TensorFlow from source.

This side project has the goal of compiling TensorFlow from source to make the best use of my laptop's computing resources. The insstructions below describe my experience in installing tensorflow on my laptop which has an Intel i5-7300hq CPU, and Nvidia 1050Ti GPU and Ubuntu 16.04.2 OS.



### [TensorFlow Installation Instructions](https://www.tensorflow.org/install/install_sources)

*Note: Link used at 16.06.2017. Information provided may be different at a later date.*

#### Perform preparing environment for Linux instructions. (Python and Java dependencies)
```{shell}
$ python3 -V
Python 3.5.2
$ java -version
openjdk version "1.8.0_131"
OpenJDK Runtime Environment (build 1.8.0_131-8u131-b11-0ubuntu1.16.04.2-b11)
OpenJDK 64-Bit Server VM (build 25.131-b11, mixed mode)
```

#### Install bazel
[Basel Website Installation Instructions](https://bazel.build/versions/master/docs/install.html) [Accessed at 16.06.2017]. We decide to use the _Bazel custom APT repository_ (which is the recommended method).

```{shell}
# Add Bazel distribution URI as a package source (one time setup)
$ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
$ curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install bazel
# Note: When installing bazel java's ibm was installed !
# The following packages were installed in the system ```ibm-java80-jdk ibm-java80-jre```
$ java -version
java version "1.8.0"
Java(TM) SE Runtime Environment (build pxa6480sr4fp1-20170215_01(SR4 FP1))
IBM J9 VM (build 2.8, JRE 1.8.0 Linux amd64-64 Compressed References 20170209_336038 (JIT enabled, AOT enabled)
J9VM - R28_20170209_0201_B336038
JIT  - tr.r14.java.green_20170125_131456
GC   - R28_20170209_0201_B336038_CMPRSS
J9CL - 20170209_336038)
JCL - 20170215_01 based on Oracle jdk8u121-b13
```
This is not intended - but will let it be for now.

#### Install python3.5 tf build  dependencies
```{shell}
$ sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel
```
Those packages were already installed on my system, so I canceled the command on the confirmation prompt.
Care has to be taken because apt will not see relevant packages when installed through pip!!

#### Optional: install TensorFlow for GPU prerequisites.
They have already been installed to get tf isntallation working.

However when double checking we cannot find the LD\_LIBRARY\_PATH enviromental variable. Hence we go through the [nvidia instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#post-installation-actions)
 again ommiting steps that have been undertaken during installation of previous tensorflow versions.

We decide to add the needed enviromental variables in a non-persistent way.
```{shell}
$ export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Invidia also has some [instructions](https://developer.nvidia.com/cudnn) for CuDNN but nothing relevant was found on this page. The site has instructions for CuDNN 7 but we have 5.1 - which
is the one used by TF. I remember installing CuDNN 5.1 after registering on the nvidia developer program.


#### Install libcupti-dev -- already installed.

```{shell}
$ sudo apt-get install libcupti-dev
```

#### Go at the TensorFlow source code repository
I have cloned the tensorflow repository. 
```{shell}
$ cd ~/tensorflow
# go to r1.2 branch
$ git checkout r1.2
```

#### Configuring the compilation:
After checking out on the 1.2.0 release branch we start the configuration script and enter the choices described below:

```{shell}
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3
Found possible Python library paths:
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.5/dist-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]
/usr/local/lib/python3.5/dist-packages
Do you wish to build TensorFlow with MKL support? [y/N] y
MKL support will be enabled for TensorFlow
Do you wish to download MKL LIB from the web? [Y/n] y
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 
Do you wish to use jemalloc as the malloc implementation? [Y/n] y
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] n
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N] n
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] n
No XLA JIT support will be enabled for TensorFlow
Do you wish to build TensorFlow with VERBS support? [y/N] n
No VERBS support will be enabled for TensorFlow
Do you wish to build TensorFlow with OpenCL support? [y/N] n
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] y
CUDA support will be enabled for TensorFlow
Do you want to use clang as CUDA compiler? [y/N] n
nvcc will be used as CUDA compiler
Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to use system default]:
Please specify the location where CUDA  toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 
Please specify the cuDNN version you want to use. [Leave empty to use system default]: 
Please specify the location where cuDNN  library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 6.1
Extracting Bazel installation...
...........
unexpected pipe read status: (error: 2): No such file or directory
Server presumed dead. Now printing '/home/nikolaos/.cache/bazel/_bazel_nikolaos/334933613be2eb2c6cb8e77893b9ff80/server/jvm.out':
java.lang.ExceptionInInitializerError
	at java.lang.J9VMInternals.ensureError(J9VMInternals.java:141)
	at java.lang.J9VMInternals.recordInitializationFailure(J9VMInternals.java:130)
	at com.google.devtools.build.lib.skyframe.SkyframeExecutor.skyFunctions(SkyframeExecutor.java:348)
	at com.google.devtools.build.lib.skyframe.SkyframeExecutor.init(SkyframeExecutor.java:586)
	at com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor.init(SequencedSkyframeExecutor.java:252)
	at com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor.create(SequencedSkyframeExecutor.java:211)
	at com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor.create(SequencedSkyframeExecutor.java:162)
	at com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutorFactory.create(SequencedSkyframeExecutorFactory.java:48)
	at com.google.devtools.build.lib.runtime.WorkspaceBuilder.build(WorkspaceBuilder.java:81)
	at com.google.devtools.build.lib.runtime.BlazeRuntime.initWorkspace(BlazeRuntime.java:204)
	at com.google.devtools.build.lib.runtime.BlazeRuntime.newRuntime(BlazeRuntime.java:1023)
	at com.google.devtools.build.lib.runtime.BlazeRuntime.createBlazeRPCServer(BlazeRuntime.java:850)
	at com.google.devtools.build.lib.runtime.BlazeRuntime.serverMain(BlazeRuntime.java:789)
	at com.google.devtools.build.lib.runtime.BlazeRuntime.main(BlazeRuntime.java:570)
	at com.google.devtools.build.lib.bazel.BazelMain.main(BazelMain.java:56)
Caused by: java.lang.ClassCastException: com.ibm.lang.management.UnixExtendedOperatingSystem incompatible with com.sun.management.OperatingSystemMXBean
	at com.google.devtools.build.lib.util.ResourceUsage.<clinit>(ResourceUsage.java:45)
	... 13 more
```

The solution to this problem is described in [this tensorflow issue](https://github.com/tensorflow/tensorflow/issues/8092).
We run
```{shell}
$ sudo apt-get install openjdk-8-jdk
```
on top of our ibm java installation. We will see later that this fixes the problem. However it leaves two installations of Java 8.0 language on our system. I am not keen to trouble shoot the issue, here are some notes towards that goal.
- Uninstalling bazel (from apt) will also uninstall ibm java. Actually uninstalling ibm's packages (ibm-java80-jre, ibm-java80-jdk) will also uninstall bazel (correctlry as they are dependencies).
- Most likely another way should have been used to install bazel.

After this we re-run the installation script:

```{shell}
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3
Found possible Python library paths:
  /usr/local/lib/python3.5/dist-packages
  /usr/lib/python3/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python3.5/dist-packages]
/usr/local/lib/python3.5/dist-packages
Do you wish to build TensorFlow with MKL support? [y/N] y
MKL support will be enabled for TensorFlow
Do you wish to download MKL LIB from the web? [Y/n] y
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 
Do you wish to use jemalloc as the malloc implementation? [Y/n] y
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] n
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N] n
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] n
No XLA JIT support will be enabled for TensorFlow
Do you wish to build TensorFlow with VERBS support? [y/N] n
No VERBS support will be enabled for TensorFlow
Do you wish to build TensorFlow with OpenCL support? [y/N] n
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] y
CUDA support will be enabled for TensorFlow
Do you want to use clang as CUDA compiler? [y/N] n
nvcc will be used as CUDA compiler
Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 8.0
Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 
Please specify the cuDNN version you want to use. [Leave empty to use system default]: 5
Please specify the location where cuDNN 5 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 6.1
........
INFO: Starting clean (this may take a while). Consider using --async if the clean takes more than several minutes.
Configuration finished
```

#### Configuration options
I didn't know what some of the configuration options where:
- [Intel Math Kernel Library (MKL)](https://en.wikipedia.org/wiki/Math_Kernel_Library). Enabled it even though I don't think it 'll contribute much because of the dedicated GPU.
- [VERBS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/verbs): This appears to be a [Remote direct memory access](https://en.wikipedia.org/wiki/Remote_direct_memory_access) feature. Since I only plan to use tf on my laptop this was disabled.

#### Build the pip package with GPU support:
Moving forward I build tensorflow:
```{shell}
$ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
```
The build process gives a huge number of wanring messages captured in [this file](./CompilationOutput.txt).

Subsequently I build the pip/wheel binary
```{shell}
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
Sat 17 Jun 15:48:42 BST 2017 : === Using tmpdir: /tmp/tmp.3WjtRls7l4
~/tensorflow/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles ~/tensorflow
~/tensorflow
/tmp/tmp.3WjtRls7l4 ~/tensorflow
Sat 17 Jun 15:48:44 BST 2017 : === Building wheel
warning: no files found matching '*.dll' under directory '*'
warning: no files found matching '*.lib' under directory '*'
~/tensorflow
Sat 17 Jun 15:48:54 BST 2017 : === Output wheel file is in: /tmp/tensorflow_pkg
```
- Comprehension Question: Why did I need to do two steps? What does the first step do if not create the binary of the compiled program? (And if so where is the output file placed?)

Finally let's install our new package!
```{shell}
$ sudo -H pip3 install /tmp/tensorflow_pkg/tensorflow-1.2.0-cp35-cp35m-linux_x86_64.whl
Processing /tmp/tensorflow_pkg/tensorflow-1.2.0-cp35-cp35m-linux_x86_64.whl
Requirement already satisfied: six>=1.10.0 in /usr/lib/python3/dist-packages (from tensorflow==1.2.0)
Collecting backports.weakref==1.0rc1 (from tensorflow==1.2.0)
  Downloading backports.weakref-1.0rc1-py3-none-any.whl
Requirement already satisfied: wheel>=0.26 in /usr/lib/python3/dist-packages (from tensorflow==1.2.0)
Collecting bleach==1.5.0 (from tensorflow==1.2.0)
  Using cached bleach-1.5.0-py2.py3-none-any.whl
Collecting markdown==2.2.0 (from tensorflow==1.2.0)
  Downloading Markdown-2.2.0.tar.gz (236kB)
    100% |████████████████████████████████| 245kB 991kB/s 
Requirement already satisfied: werkzeug>=0.11.10 in /usr/local/lib/python3.5/dist-packages (from tensorflow==1.2.0)
Requirement already satisfied: protobuf>=3.2.0 in /usr/local/lib/python3.5/dist-packages (from tensorflow==1.2.0)
Requirement already satisfied: numpy>=1.11.0 in /usr/local/lib/python3.5/dist-packages (from tensorflow==1.2.0)
Collecting html5lib==0.9999999 (from tensorflow==1.2.0)
  Downloading html5lib-0.9999999.tar.gz (889kB)
    100% |████████████████████████████████| 890kB 687kB/s 
Requirement already satisfied: setuptools in /usr/local/lib/python3.5/dist-packages (from protobuf>=3.2.0->tensorflow==1.2.0)
Requirement already satisfied: packaging>=16.8 in /usr/local/lib/python3.5/dist-packages (from setuptools->protobuf>=3.2.0->tensorflow==1.2.0)
Requirement already satisfied: appdirs>=1.4.0 in /usr/local/lib/python3.5/dist-packages (from setuptools->protobuf>=3.2.0->tensorflow==1.2.0)
Requirement already satisfied: pyparsing in /usr/local/lib/python3.5/dist-packages (from packaging>=16.8->setuptools->protobuf>=3.2.0->tensorflow==1.2.0)
Building wheels for collected packages: markdown, html5lib
  Running setup.py bdist_wheel for markdown ... done
  Stored in directory: /root/.cache/pip/wheels/b9/4f/6c/f4c1c5207c1d0eeaaf7005f7f736620c6ded6617c9d9b94096
  Running setup.py bdist_wheel for html5lib ... done
  Stored in directory: /root/.cache/pip/wheels/6f/85/6c/56b8e1292c6214c4eb73b9dda50f53e8e977bf65989373c962
Successfully built markdown html5lib
Installing collected packages: backports.weakref, html5lib, bleach, markdown, tensorflow
  Found existing installation: html5lib 0.999999999
    Uninstalling html5lib-0.999999999:
      Successfully uninstalled html5lib-0.999999999
  Found existing installation: bleach 2.0.0
    Uninstalling bleach-2.0.0:
      Successfully uninstalled bleach-2.0.0
  Found existing installation: Markdown 2.6.8
    Uninstalling Markdown-2.6.8:
      Successfully uninstalled Markdown-2.6.8
Successfully installed backports.weakref-1.0rc1 bleach-1.5.0 html5lib-0.9999999 markdown-2.2.0 tensorflow-1.2.0
```

#### Testing the new installation
We test the new installation on the terminal:
```{shell}
$ python3
Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version__)
1.2.0
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
2017-06-17 15:53:22.272460: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-06-17 15:53:22.273181: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties: 
name: GeForce GTX 1050 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.62
pciBusID 0000:01:00.0
Total memory: 3.94GiB
Free memory: 3.53GiB
2017-06-17 15:53:22.273219: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0 
2017-06-17 15:53:22.273231: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y 
2017-06-17 15:53:22.273261: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0)
>>> print(sess.run(hello))
b'Hello, TensorFlow!'
>>> exit()
```

When testing installed pip packages we find a strange thing:
```{shell}
$ pip3 show tensorflow
Name: tensorflow
Version: 1.2.0
Summary: TensorFlow helps the tensors flow
Home-page: http://tensorflow.org/
Author: Google Inc.
Author-email: opensource@google.com
License: Apache 2.0
Location: /usr/local/lib/python3.5/dist-packages
Requires: html5lib, werkzeug, wheel, six, protobuf, numpy, bleach, backports.weakref, markdown
$ pip3 show tensorflow-gpu
Name: tensorflow-gpu
Version: 1.1.0
Summary: TensorFlow helps the tensors flow
Home-page: http://tensorflow.org/
Author: Google Inc.
Author-email: opensource@google.com
License: Apache 2.0
Location: /usr/local/lib/python3.5/dist-packages
Requires: numpy, six, werkzeug, wheel, protobuf
```
The old package hasn't been removed!
This probably has to do with the different names of the packages (tensorflow vs tensorflow-gpu).
We try uninstalling ourselves:
```{shell}
$ sudo -H pip3 uninstall tensorflow-gpu
[sudo] password for $USER: 
Uninstalling tensorflow-gpu-1.1.0:
...
Proceed (y/n)? y
  Successfully uninstalled tensorflow-gpu-1.1.0
```

This appeared to break tensorflow, but it may have been that I was trying to run it from the tensorflow source repository.
I have subsequently uninstalled it and re-installed it from source. Therefore it is suggested to uninstall the previous version of
tensorflow before installing the new one.

According to [this stack overflow answer](https://stackoverflow.com/a/35963479/1904901)
starting python and then tensorflow within the tensorflow source code repository may cause problems !
