import torch
from torch._C import _jit_get_all_schemas


def list_all_node_kinds():
    schemas = _jit_get_all_schemas()
    kinds = set()
    for schema in schemas:
        kinds.add(schema.name)

    for kind in sorted(kinds):
        print(kind)


if __name__ == "__main__":
    list_all_node_kinds()
