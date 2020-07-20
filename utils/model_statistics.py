""" model_statistics.py """
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch

from .formatting import FormattingOptions, Verbosity
from .layer_info import LayerInfo

HEADER_TITLES = {
    "kernel_size": "Kernel Shape",
    "output_size": "Output Shape",
    "num_params": "Param #",
    "mult_adds": "Mult-Adds",
}
CORRECTED_INPUT_SIZE_TYPE = List[Union[Sequence[Any], torch.Size]]


class ModelStatistics:
    """ Class for storing results of the summary. """

    def __init__(
        self,
        summary_list: List[LayerInfo],
        input_size: CORRECTED_INPUT_SIZE_TYPE,
        formatting: FormattingOptions,
        ucfg:{},
    ):
        self.summary_list = summary_list
        self.input_size = input_size
        self.total_input = sum([abs(np.prod(sz)) for sz in input_size])
        self.formatting = formatting
        self.total_params, self.trainable_params = 0, 0
        self.total_output, self.total_mult_adds = 0, 0
        self.bs = ucfg['batchsize']*ucfg['BPE'] #input batch size and BPE


    @staticmethod
    def to_bytes(num: int) -> float:
        """ Converts a number (assume floats, 4 bytes each) to megabytes. """
        assert num >= 0
        return num * 4 / (1024 ** 2)

    @staticmethod
    def to_readable(num: int) -> float:
        """ Converts a number to millions or billions. """
        assert num >= 0
        if num >= 1e9:
            return num / 1e9
        return num / 1e6

    def __repr__(self) -> str:
        """ Print results of the summary. """
        #header_row = self.formatting.format_row("Layer (type:depth-idx)", HEADER_TITLES)
        layer_rows = self.layers_to_str()
       
        summary_str = ("{}".format(layer_rows))
        return summary_str

    def layer_info_to_row(self, layer_info: LayerInfo, reached_max_depth: bool = False) -> str:
        """ Convert layer_info to string representation of a row. """

        def get_start_str(depth: int) -> str:
            return "├─" if depth == 1 else "|    " * (depth - 1) + "└─"
        
        def get_start_comma(depth: int) -> str: #0615
            return "" if depth == 1 else "," * (depth - 1) 

        row_values = {
            "input_size": layer_info.input_size[1:] if len(layer_info.input_size)==4 else layer_info.input_size[1:]+(['']*(4-len(layer_info.input_size))), #0614, multiple in?
            "output_size": layer_info.output_size[1:] if len(layer_info.output_size)==4 else layer_info.output_size[1:]+(['']*(4-len(layer_info.output_size))),
            "num_in": np.prod(layer_info.input_size[1:])*self.bs,
            "num_out": np.prod(layer_info.output_size[1:])*self.bs,
            "num_params": layer_info.num_params*self.bs if layer_info.num_params else '', #0615
            "mult_adds": layer_info.macs if layer_info.macs else '', #0615            
            "kernel_size": layer_info.kernel_size[2:] if len(layer_info.kernel_size)>2 else ['',''],
            "pad_size": layer_info.pad_size if layer_info.pad_size else ['',''], #0614
            "stride_size": layer_info.stride_size if layer_info.stride_size else ['',''], #0614
            "gemm": layer_info.gemm if layer_info.gemm else [''],
            "vect": layer_info.vect if layer_info.vect else [''],
            "acti":layer_info.acti if layer_info.acti else [''],
        }  #0615: list instead of string
        depth = layer_info.depth
        if self.formatting.use_branching==1: # 0615:for 3 cases
            name = get_start_str(depth) + str(layer_info) 
        elif self.formatting.use_branching==2:
            name = get_start_comma(depth) + str(layer_info) + "," * (self.formatting.max_depth-depth)
        else:
            name ='' + str(layer_info)# 0615
        # if name.find('Linear')>=0:
        #     print(name)
        new_line = self.formatting.format_row(name, row_values)
        if self.formatting.verbose == Verbosity.VERBOSE.value:
            for inner_name, inner_shape in layer_info.inner_layers.items():
                prefix = get_start_str(depth + 1) if self.formatting.use_branching else "  "
                extra_row_values = {"kernel_size": str(inner_shape)}
                new_line += self.formatting.format_row(prefix + inner_name, extra_row_values)
        return new_line

    def layers_to_str(self) -> str:
        """ Print each layer of the model as tree or as a list. """
        if self.formatting.use_branching:
            return self._layer_tree_to_str()

        layer_rows = ""
        for layer_info in self.summary_list:
            layer_rows += self.layer_info_to_row(layer_info)
        return layer_rows

    def _layer_tree_to_str(self, left: int = 0, right: Optional[int] = None, depth: int = 1) -> str:
        """ Print each layer of the model using a fancy branching diagram. """
        if depth > self.formatting.max_depth:
            return ""
        new_left = left - 1
        new_str = ""
        if right is None:
            right = len(self.summary_list)
        for i in range(left, right):
            layer_info = self.summary_list[i]
            #print(depth,layer_info.depth)
            if layer_info.depth == depth:
                reached_max_depth = depth == self.formatting.max_depth
                new_str += self.layer_info_to_row(layer_info, reached_max_depth)
                new_str += self._layer_tree_to_str(new_left + 1, i, depth + 1)
                new_left = i
        return new_str