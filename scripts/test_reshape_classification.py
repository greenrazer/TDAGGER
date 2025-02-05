def get_prefix_products(shape):
    """Calculate prefix products for a shape"""
    result = [1]
    for dim in shape:
        result.append(result[-1] * dim)
    return set(result)

def classify_reshape(shape1, shape2):
    """
    Classify a tensor reshape operation as 'group', 'ungroup', or 'neither'.
    Uses prefix product set intersection to identify valid grouping/ungrouping.
    """
    prods1 = get_prefix_products(shape1)
    prods2 = get_prefix_products(shape2)
    
    if max(prods1) != max(prods2):
        raise Exception("cannot reshape")
    
    if prods2.issubset(prods1):
        return "group"
    elif prods1.issubset(prods2):
        return "ungroup"
    else:
        return "neither"

def get_reshape_mapping(shape1, shape2):
    """
    Get detailed axis mapping for a reshape operation.
    
    For group operations: Returns list where each element is either an int or list of ints,
        indicating which axes were grouped together.
    For ungroup operations: Returns list where each element is either an int or tuple(int, list),
        indicating which axes an original dimension was split into.
    
    Args:
        shape1: Original shape (tuple of integers)
        shape2: New shape (tuple of integers)
        
    Returns:
        List of mappings, or None if the reshape is invalid
    """
    op_type = classify_reshape(shape1, shape2)
    if op_type == "neither":
        return None
        
    def get_cumulative_products(shape):
        """Calculate cumulative products with indices"""
        result = [(1, [])]  # (product, indices that made this product)
        curr_indices = []
        for i, dim in enumerate(shape):
            curr_indices.append(i)
            prev_prod = result[-1][0]
            result.append((prev_prod * dim, curr_indices[:]))
        return result
    
    if op_type == "group":
        # source_prods = get_cumulative_products(shape1)
        target_prods = get_cumulative_products(shape2)
        
        mapping = []
        src_idx = 0
        
        for _, _ in target_prods[1:]:  # Skip the initial 1
            # Find how many source dims we need to multiply
            group = []
            curr_prod = 1
            while curr_prod != shape2[len(mapping)]:
                group.append(src_idx)
                curr_prod *= shape1[src_idx]
                src_idx += 1
            mapping.append(group if len(group) > 1 else group[0])
            
        return mapping
        
    else:  # ungroup
        mapping = []
        tgt_idx = 0
        
        for src_idx, src_dim in enumerate(shape1):
            # Find factors that multiply to make this dimension
            factors = []
            curr_prod = 1
            while curr_prod != src_dim:
                factors.append(shape2[tgt_idx])
                curr_prod *= shape2[tgt_idx]
                tgt_idx += 1
            mapping.append((src_idx, factors) if len(factors) > 1 else src_idx)
            
        return mapping

# Test cases
test_cases = [
    ((2, 3, 4), (6, 4)),           # group: [[0, 1], 2]
    ((6, 4), (2, 3, 4)),           # ungroup: [(0, [2, 3]), 1]
    ((2, 3, 4, 5), (6, 20)),       # group: [[0, 1], [2, 3]]
    ((24, 5), (2, 3, 4, 5)),       # ungroup: [(0, [2, 3, 4]), 1]
    ((2, 2, 3), (4, 3)),           # group: [[0, 1], 2]
    ((4, 3), (2, 2, 3)),           # ungroup: [(0, [2, 2]), 1]
    ((12, 4), (2, 2, 3, 4)),       # ungroup: [(0, [2, 2, 3]), 1]
    ((2,3,4), (4, 6)),              # neither
    ((6,7,8), (1,1,1,2,3,1,1,1,7,1,1,8)),
    ((1,1,1,2,3,1,1,1,7,1,1,8), (6,7,8)),
    ((24,30,7), (2,60,42))
]

for shape1, shape2 in test_cases:
    op_type = classify_reshape(shape1, shape2)
    if op_type != "neither":
        result = get_reshape_mapping(shape1, shape2)
        print(f"{shape1} -> {shape2}: {op_type}{result}")
    else:
        print(f"{shape1} -> {shape2}: {op_type}")