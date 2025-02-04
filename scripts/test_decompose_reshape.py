import math
from itertools import permutations

def decompose_reshape(input_shape, output_shape):
    def prime_factors(n):
        factors = []
        i = 2
        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n = n // i
            i += 1
        if n > 1:
            factors.append(n)
        return tuple(factors)

    def flatten(arr):
        return [i for g in arr for i in g ]
    
    def prod(arr):
        out = 1
        for a in arr:
            out *= a
        return out

    if prod(input_shape) != prod(output_shape):
        print(prod(input_shape), prod(output_shape))
        raise Exception("cannot reshape")
    
    input_groups = [prime_factors(d) for d in input_shape]
    output_groups = [prime_factors(d) for d in output_shape]

    print("prime factors")

    def lazy_product(groups):  # Take original groups instead of iterators
        if not groups:
            yield ()
            return
            
        first_group = groups[0]
        rest = groups[1:]
        
        for perm in permutations(first_group):
            for rest_combo in lazy_product(rest):
                yield (perm,) + rest_combo

    for input_perm_combo in lazy_product(input_groups):
        
        for output_perm_combo in lazy_product(output_groups):
            print([1 if x == y else 0 for x,y in zip(flatten(input_perm_combo), flatten(output_perm_combo))])
            if flatten(input_perm_combo) == flatten(output_perm_combo):
                return input_perm_combo, output_perm_combo
                
    raise Exception("No reshape decomposition found")

# Test cases
print("Test 1: Common dimensions")
input_shape = (4, 32, 32, 4)
output_shape = (4, 16, 16, 16)
ops = decompose_reshape((4, 32, 32, 4), (4, 16, 16, 16))
op1_ref = "(4, (2,2,2,2,2), (2,2,2,2,2), (2,2))"
op2_ref = "(4, (2,2,2,2), (2,2,2,2), (2,2,2,2))"
print("input_shape", input_shape)
print("ops 0", ops[0], "expected:", op1_ref)
print("ops 1", ops[1], "expected:", op2_ref)
print("output_shape", output_shape)

print("\nTest 2: No common dimensions") 
input_shape = (4, 6, 10, 8)
output_shape = (12, 20, 8)
ops = decompose_reshape((4, 6, 10, 8), (12, 20, 8))
op1_ref = "((2,2), (3,2), (2,5), 8)"
op2_ref = "((2,2,3), (2,2,5), 8)"
print("input_shape", input_shape)
print("ops 0", ops[0], "expected:", op1_ref)
print("ops 1", ops[1], "expected:", op2_ref)
print("output_shape", output_shape)

print("\nTest 3: Ambiguous dimension swapping")
input_shape = (4, 6, 10, 8)
output_shape = (8, 30, 8)
ops = decompose_reshape(input_shape, output_shape)
op1_ref = "((2,2), (2,3), (2,5), 8)"
op2_ref = "((2,2,2), (3,2,5), 8)"
print("input_shape", input_shape)
print("ops 0", ops[0], "expected:", op1_ref)
print("ops 1", ops[1], "expected:", op2_ref)
print("output_shape", output_shape)

print("\nTest 4: Mixed common dimensions")
input_shape = (2, 8, 4, 16)
output_shape = (4, 2, 16, 8)
ops = decompose_reshape(input_shape, output_shape)
op1_ref = "(2, (2,2,2), (2,2), (2,2,2,2))"
op2_ref = "((2,2), 2, (2,2,2,2), (2,2,2))"
print("input_shape", input_shape)
print("ops 0", ops[0], "expected:", op1_ref)
print("ops 1", ops[1], "expected:", op2_ref)
print("output_shape", output_shape)


print("\nTest 5: Longgggg")
input_shape = (1_500_000, 2, 8, 4, 16, 10, 4, 16, 3, 6, 2)
output_shape = (100_000, 15, 4, 2, 16, 8, 10, 4, 16, 3, 6, 2)
ops = decompose_reshape(input_shape, output_shape)
op1_ref = "(2, (2,2,2), (2,2), (2,2,2,2))"
op2_ref = "((2,2), 2, (2,2,2,2), (2,2,2))"
print("input_shape", input_shape)
print("ops 0", ops[0], "expected:", op1_ref)
print("ops 1", ops[1], "expected:", op2_ref)
print("output_shape", output_shape)