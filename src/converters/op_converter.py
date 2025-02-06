from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Tuple, TypeVar

# Generic type variables
ContextT = TypeVar("ContextT")  # Type of the conversion context
ConverterT = TypeVar("ConverterT")  # Type of the converter functions
InputT = TypeVar("InputT")  # Type of the primary input (torch_op or op)
OutputT = TypeVar("OutputT")  # Type of the conversion output


class OpConverter(ABC, Generic[ContextT, ConverterT, InputT, OutputT]):
    """
    Base class for operation converters that handle translation between different formats.

    Generic Parameters:
        ContextT: The type of the conversion context
        ConverterT: The type of the converter functions
        InputT: The type of the op to convert
        OutputT: The type of the conversion output
    """

    def __init__(self):
        self._converters: Dict[str, ConverterT] = {}
        self._register_converters()

    @abstractmethod
    def _register_converters(self) -> None:
        """Register all converter functions in the _converters dictionary."""
        pass

    @abstractmethod
    def _create_context(self, *args, **kwargs) -> ContextT:
        """Create a context object from the provided arguments."""
        pass

    @abstractmethod
    def _get_operation_key(self, input_value: InputT) -> str:
        """Extract the key used to look up the appropriate converter."""
        pass

    def convert_op(self, op: InputT, *args, **kwargs) -> OutputT:
        """
        Convert an operation using the registered converters.

        Args:
            op: The operation to convert
            *args: Additional arguments needed for context creation
            **kwargs: Additional keyword arguments needed for context creation

        Returns:
            The converted operation

        This template method implements the common conversion flow:
        1. Create context from arguments
        2. Get operation key
        3. Validate operation type
        4. Execute conversion
        """
        # Create context
        ctx = self._create_context(op, *args, **kwargs)

        # Get operation key for converter lookup
        op_key = self._get_operation_key(op)

        # Validate operation type
        if op_key not in self._converters:
            raise Exception(f"Unsupported operation type: {op_key}")

        # Execute conversion
        return self._converters[op_key](ctx)
