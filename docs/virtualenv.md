#  Install Virtual Environment

​	It's recommended to run the tool on a virtual environment. Installation using Anaconda is demonstrated below. 	

## 1. Install Anaconda

1.1. Download the latest 'anaconda' compatible  to the OS, and install it

​		The package can be downloaded from the [official site](https://www.anaconda.com/products/individual) or a [mirror site](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

1.2. Set a local mirror source (optional)

​		If it is inconvenient to access the official site,  a local mirror source can be added. For example, the mirror site from colleges,

​		https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

​		One can set the source channel by adding the source links into '.condarc' which is under the system's user folder. For windows OS, the file is usually at 'C:\Users\xxx', where 'xxx' is the login user name.



## 2. Create a virtual environment 

Launch the anaconda command windows, in the window, 

2.1. Create a environment by:

​			`conda create nnp`

2.2. enter the environment

​			`conda activate nnp`

2.3. install the python package 

​			`conda install xx` 

​		where xx is the package name. The following packages are required for the tool:

​		Python 3.6+, 	Tensorflow 2.1 +,  Pytorch 1.5+

​		pandas,​		numpy,​		matplotlib
​		openpyxl, 	scikit-learn, 	scikit-image
​		graphviz, 	python-graphviz, 		pydot

2.4 other packages

​	pip tool can be used to install packages under the activated virtual environment. For example,

​			`pip3 install keras-bert`

​	Similar to the conda, a [pip mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) can be used to accelerate the installation.

