#  Machine Learning Parser

​	The tool was built for AI hardware/ software architectures to quickly understand and analyze the overall and layer-wise structure of neural network models. 

​	The types of results will be generated by the tool:

​	1) **tables** of parameters & statistics of network structure in excel format

​	2)  **graphs** of model structures in pdf format

## 1. Installation:

It is recommended to install the tool in a virtual environment, explained in [the manual](./docs/virtualenv.md). 

1.1. clone the repository to the local drive. 

1.2. The tool works with :

​		Python 3.6+, 

​		Tensorflow 2.1 +, 

​		Pytorch 1.5+

1.3. Install the following Python modules:

​		pandas
​		numpy
​		matplotlib
​		openpyxl
​		scikit-learn
​		scikit-image
​		graphviz
​		python-graphviz
​		pydot



## 2 Usages:

​	There are two versions of tools integrated in the package, one for ML framework pytorch: *torch2table*; the other is for Tensorflow via keras: *keras2table*.

#### 2.1. Pytorch version: torch2Table

##### 2.1.1. Command

- type the command as the following format to get the results

  ​	`python torch2table.py -n resnet50 -b 1 -e 1` 

- three optional arguments are:

  -n: the name of the neural network model. Tested models are listed below. Please note that the name is **case-sensitive**

  -b: batch-size of the input data

  -e: element size in byte, for example, float32 = 4, int8 =1, etc.

- The results are exported as the excel tables in ''/output/torch/''. 

##### 2.1.2. Tested models:

1. torchvision: 

   - base models

   ​	alexnet, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg_13 bn, vgg16_bn,

   ​	vgg19_bn, resnet18, resnet34, resnet50, resnet101, resnet152, 

   ​	SqueezeNet1_0, SqueezeNet1_1, densenet_121, densenet_169, densenet_201

   ​	densenet_161, inception_v3, googlenet, shufflenet_V2_x'n'_'n', mobileNet_v2

   ​	resnext_50_32x4d, resnext_101_32x8d, wide_resNet_50_2, wide_resnet_101_2

   ​	MNASNet'n'_'n'

   - detection model

   ​	maskrcnn 

2. Recomendation: 

   ​	dlrm

3. RNN network: 

   - base: lstm, gru
   - gnmt

4.  one stage detection: ssd_mobilenet, ssd_r34

5. Others Models can also be imported, as long as a input tensor is provided  with the model. Please refer to **Section 4** for details.

### 2.2. Tensorflow version: keras2table

##### 2.2.1. Command

- type the command as the following format to get the results

  ​	`python keras2table.py -n ResNet50 -b 1 -e 1` 

- three optional arguments are:

  -n: the name of the neural network model. Tested models are listed below. Please note that the name is **case-sensitive**

  -b: batch-size of the input data

  -e: element size in byte, for example, float32 = 4, int8 =1, etc.

- The results are exported as the excel tables in ''/output/tf/''

##### 2.2.2. Tested models:

1. keras pretrianed models: 

​    'DenseNet121', 'DenseNet169', 'DenseNet201',

​    'InceptionResNetV2', 'InceptionV3',

​     'MobileNet', 'MobileNetV2',

​     'NASNetLarge', 'NASNetMobile',

​     'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2',

​     'VGG16', 'VGG19',

​     'Xception',

2. Reomendeation: din

3. EfficientNet: EfficientNetB0 ~ EfficientNetB7

4. NLP: bert

##### 2.2.3. Notes:

- keras-bert module should be installed for analysis,

    ​		`pip3 install keras-bert` 

- For DIN model, please clone the packages from [tensorflow/models](https://github.com/tensorflow/models) to a local folder, and add the folder into  PYTHONPATH



## 3. Outputs

​	There are two types of output files in the //outputs// torch// ( or //tf) folder: tables in xlsx and graph in pdf. Both are named as the input name of neural networks. 

### 3.1. The excel table with two sheets.  

#### 	3.1.1. summaries sheet:

​	 	Total counts of memory and computation costs. 

​			Note that  1M = 1024 x 1024 for data  size and 1G = 1E9 for ops counts

#### 	3.1.2. Details sheet:

​		The results are demonstrated at one nn layer per row. The meanings of columns as below:

​		**Layer:**

​			**TF Keras**: Layer names & Types

​			**Pytorch**:  layer names in multi-levels.  Pytorch models are organized in a nested style. For example，a model may have several sequential/ sub-modules, and each module also have several nn layers.  *The first columns* in the  table demonstrate the hierarchical structures: the layer names in the first columns are at the top level of the model structure, the layer names in second column are at the second level of the model, etc. , as shown below,

```
                                                            layer-l0		|	layer_l1	|	layer-l1
       	├─Sequential: 1-5                                  Sequential:1-5	|			|
        |    └─Bottleneck: 2-1                           			| Bottleneck:2-1	|
        |    |    └─Conv2d: 3-1             ==>				     	|		        | Conv2d: 3-1
        |    |    └─BatchNorm2d: 3-2             			        |		        | BatchNorm2d: 3-2
        |    |    └─ReLU: 3-3      						|		        | ReLU: 3-3
        
        Pytoch Model Hierarchy								pytorch table
```

​		**Input tensors :**

​			TF Keras version: 

​				I0_1,I0_2, I0_3: the shape of the first input

​				I1_1,I1_2,I1_3: the shape of the second input

​			Pytorch version:

​				I1, I2, I3: the shape of the first tensor

​		**Output tensors:** O1,O2,O3: th eshap of the first output

​		**Kernel Tensors:** 

​			k1,k2: kernel size H&W for conv & pooling; 

​			s1,s2: stride H&W of rolling windows;

​			p1,p2: padding size, values are calculated based on centric padding

​		**Memory Costs:** 

​			SizeI: Size of input tensors

​			SizeO: Size of output tensors

​			SizeW: Size of model parameters，note that the values depends on Byte per Elements(BPE) of the models, default BPE is 4 (fp32)

​		**Computation Costs:**

​			OpGemm: # of Matrix multi-adds

​			OpVect: # of Element-wise Ops

​			OpActi: # of activations(for transcendental Functions). Relu is also counted as an activation operation for convenience.

​		**misc:**

​			additional input/output tensors of the nn layer

​		**Color bars** of extreme cells

​			The cells/layers with the maximum  cost are marked as:

​				The maximum output Tensor:  Red,

​				The maximum weight tensor: Pink

​				The maximum matrix multi-add  costs : Green.

### 3.2  Model Graphs

​	A graph to visualize the model structure will be also generated with the same name as the table file.



## 4. Add your own model

​	One can simply leverage the API in 'newmodel.py' in the root to analyze any customized models, as long as the models are built-up and ready for test. 

​	There are two major steps for a new model:  1 set the  model info ; 2 execute the main parser codes. 

#### 4.1 pytorch

##### 	a. Set the model info

​		To parse the nn model, all required configures should be provided in function 'pymodel()' in 'newmodel.py'.

​		- model_path: the absolute path which contains the 'model_file'

​		- load model:     from 'model_file' import 'your_model'

​									model = your_model(args)    # args: parameters of your model

​		- define inputs of the model:  x = torch.rand(1,3, 300, 300)   #  tensor in (BCHW) format

​						you may define the input tensors for the customized model.  Multiple inputs are formated as a list of tensors, as shown in the demo.

#####   b. execute the parser

​	 type the command as the following format to get the results:

​				`python torch2table.py -n newmodel  --model your_model`

  The optional argument --model is to get the model name. If the no model name provided, the results of the model will be saved as ' newmodel.xlsx' in the outputs folder.

### 4.2 keras-tf

​	The operations for Keras model are similar to pytorch model: 

​		1 set the model configs. in 'tfmodel()' in 'newmodel.py' ; 

​		2 execute command 

​				`python keras2table.py -n newmodel  --model your_model`



## 5. Advances

​	One can further analyze the various configures of a NN models by changing default settings in the codes. Two examples are shown below.   

​	Note that the functions ( getmodel() & modelLst() ) mentioned below also demonstrate the ways to add a neural network model. Additional neural network models can be added using the similar approaches. 

### Example 1: bert in keras-tf

1. Bert models with different settings  

   The setting of bert model can be obtained by changing the optional arguments of  the 'get_model' function at line 320 in //utils//tftools//getmodel(), please refer to the [keras-bert](https://pypi.org/project/keras-bert/) for detailed settings for the bert model

2. inference model

   To get bert inference model, two revisions are required: 

   1. change training = Flase at line 310  //utils//tftools//getmodel()

   2. revise  `return inputs, transformed` -> `return inputs, transformed,model` at the orignal scirpts at line164 at \\\lib\\\site-packages\\\keras_bert\\\bert.py. Note that the operation will change the original codes of keras-bert

### Example 2:  dlrm in pytorch

1. Different settings of dlrm model

   The setting of dlrm model can be adjusted by changing values of variables at line62-72 in  in //utils//pytools//modellst(). Please refer to [DLRM](https://github.com/facebookresearch/dlrm) for the details of these variables.



​	Please note that special network structures and customized operators may not be supported by the tool. Feel free to join us to extend the functions of the tool.

### References:

- Codes for Pytorch model estimation were revised based on [torch-summary](https://github.com/TylerYep/torch-summary) @TylerYep
- Keras version EfficientNet is originated from [EfficientNet](https://github.com/qubvel/efficientnet/tree/master/efficientnet) @qubvel  
- DLRM is originated from [DLRM](https://github.com/facebookresearch/dlrm) by facebook
- ML Perf models from [ML perf](https://github.com/mlperf) @mlperf
