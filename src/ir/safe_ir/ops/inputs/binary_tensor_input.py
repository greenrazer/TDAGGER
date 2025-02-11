from typing import List

from .op_input import OpInput


class BinaryTensorInput(OpInput):
    input_0: str
    input_1: str

    def __init__(self, input_0: str, input_1: str):
        inputs = {"input_0": input_0, "input_1": input_1}
        super().__init__(inputs)
        self.input_0 = input_0
        self.input_1 = input_1
