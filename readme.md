#  Machine Learning Parser

The tools were built for AI hardware/ software architecture to understand and analyze the lay-wise structure of neural network models. The results are tables of parameters of network structure, one layer per row, in csv format. The tables can be  utilized for further performance analysis. 

### Pytorch version: torch2csv

The scripts is to parse the neural network models in the pytorch framework. 

The nn models has been tested are listed at the beginning.  Set the variable 'nnname' as desired model name to produce the table. The result is in ''/output/torch/''

Note:

For transformer models, [Huggingface's Transformer package](https://github.com/huggingface/transformers) should be installed.



### Tensorflow version: keras2csv

Tensorflow version  tool can parse the TF models using Keras API on Tensorflow framework version >2.0

The nn models has been tested are listed at the beginning.  Set the variable 'nnname' as desired model name to produce the table. The result is in ''/output/tf/''