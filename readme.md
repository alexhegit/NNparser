#  Machine Learning Parser

The tools were built for AI hardware/ software architecture to quickly understand and analyze the lay-wise structure of neural network models. The results are tables of parameters of network structure, one layer per row, in excel format.  

### Installation:

1 clone the repository to the local drive. 

2 Besides ML frameworks, Python modules pandas, numpy, opnpyxl, scikit-learn, matplotlib may have to be installed



### Usages:

### Pytorch version: torch2csv

- The scripts is to parse the neural network models in the pytorch framework. 

- Set the variable 'nnname' as desired model name to produce the table.  Names of the tested nn models   are listed at the beginning.  

- The default settings of NN models and input tensors can be adjusted manually.

- The results are exported as the excel tables in ''/output/torch/''. 

##### Note:

- For transformer models, [Huggingface's Transformer package](https://github.com/huggingface/transformers) should be installed.

- For DIEN/NCF models, packages from [tensorflow/models](https://github.com/tensorflow/models) should be included. A easier solution is to clone the codes to a folder, and add the folder into PYTHONPATH



### Tensorflow version: keras2csv

Tensorflow version  tool can parse the TF models using Keras API on Tensorflow framework version >2.0

Names of the tested nn models are listed at the beginning.  Set the variable 'nnname' as desired model name to produce the table. 

The results are exported as the excel tables in ''/output/tf/''



### Outputs

The outputs are formatted tables with two sheets.  

**1 summaries:**

​	 Total counts of memory and computation costs

**2 Details:**

​	The results are demonstrated at one nn layer per row. The meanings of columns as belows:

​	**Layer:**

​		TF Keras: Layer names & Types

​		Pytorch:  layer names in multi-levels

​	**Input tensors :**

​		TF Keras version: 

​			I0_1,I0_2, I0_3: the shape of the first input

​			I1_1,I1_2,I1_3: the shape of the second input

​		Pytorch version:

​			I1, I2, I3: the shape of the first tensor

​	**Output tensors:** O1,O2,O3: th eshap of the first output

​	**Kernel Tensors:** 

​		k1,k2: kernel size H&W for conv & pooling; 

​		s1,s2: stride H&W of rolling windows;

​		p1,p2: padding size, values are calculated based on centric padding

​	**Memory Costs:** 

​		SizeI: Size of input tensors

​		SizeO: Size of output tensors

​		SizeW: Size of model papramters

​	**Computation Costs:**

​		OpGemm: # of Matrix multi-adds

​		OpVect: # of Element-wise Ops

​		OpActi: # of activations(for transcendental Functions). Relu is also counted as an activation operation for convenience.

​	**Color bars** of extreme cells

​		The cells/layers with the maximum  cost are marked as:

​			The maximum output Tensor:  Red,

​			The maximum weight tensor: Pink

​			The maximum matrix multi-add  costs : Green.

