from typing import List

from .op_input import OpInput


class UnaryTensorInput(OpInput):
    input: str

    def __init__(self, input: str):
        inputs = {"input": input}
        super().__init__(inputs)
        self.input = input
