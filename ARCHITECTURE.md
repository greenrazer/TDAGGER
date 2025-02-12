# Architecture

> [!WARNING]
> This is a work in progress

## Problems

- It is hard to tell things about models automatically, like total_flops, memory_reads, memory_writes
- It is hard to remove batch dimensions from models when needed.

## Goal

- Split the model into a series of Tensor Directed Acyclic Graphs(TDAGs)(no conditional branching, dynamic-length loops, ...)
- TDAGs have very few core operations
- Each TDAG can be thought of as a "jit.trace" of a neural network.
- TDAGs input is a list of tensors
- TDAGs output a list of tensors
- Connect TDAGs via control flow only at the top level (conditional branching, dynamic-length loops, ...)
- The top level can be thought of as a list of different traces each gated by a giant switch/match statement that decides whether 
  to pass the ouput into the switch/match statement again or to ouptut it 

## Tensor Types

- input: defined at runtime
- parameter: learnable parameter
- buffer: a non-learnable parameter that can change at runtime
- const : a non-learnable parameter that cannot change at runtime

## Op
- Each Op has inputs and a spec
- inputs contain names of ops and tensors in the graph and are 

## Core Tensor Ops
- permute
  - permute dimensions
  - "b c h w -> b h w c"
  - `torch.permute(x, [3,2,1,0])`
- group
  - basically reshape each dimension by multiplying 2 dimensions together
  - $a b c d - > a b (c d)$ 
  - `torch.reshape(x, (a, b, c * d))`
- ungroup
  - basically reshape each dimension by factoring out dimensions, can have one wild card 
  - $n c (img, 10) - > n c img 10$ 
  - `torch.reshape(n, c, -1, 10)`
- squeeze
  - helper function for group, remove a dimension where size equals one
  - $ a b 1 c d 1 - > a b c d$ 
  - `torch.squeeze(x, dim=(2, 5))`
- unsqueeze
  - helper function for ungroup, insert a dimension where size equals one
  - $ a b c d - > a b 1 c d 1$ 
  - `torch.unsqueeze(x, dim=(2, 5))`
- slice
  - same as numpy without step (except I prefer inclusive indices for my slicing)
  - `x[..., 1:, :, 1:, :1, 1:2, -1:, :-1]`
- pad
  - pad the input
  - a b -> 1 <- a -> 1 2 <- c -> 0
  - `torch.nn.functional.pad(x, (1, 1, 2, 0))`
- reduce
  - do an operation across the entire axis that reduces it to a size of s
  - $n {c: 1} h w -> n 1 h w | mean$
  - `torch.sum(x, dim=(1,3), keep_dims=True)`
- select
  - helper function for reduce to an index
  - $n {c: 1} h w -> n 1 h w$
  - `torch.unsqueeze(x[:, 1, :, :], 1)`
- repeat
  - do an operation that repeats the dim of size 1
  - $(1) (1) h w -> (3) (10) h w$
- fold
  - n c h*a w*b -> n c {h: a 3} {w: b 3}
-  `x = F.fold(x, output_size=(h, w), kernel_size=(a, b), stride=3, dilation=0, padding=0)`
- unfold
  - n c {h: a 3} {w: b 3} -> n c h*a w*b
  - sort of `x = F.unfold(pad(x), kernel_size=(a, b), stride=3, dilation=0, padding=0)`
- unary ops
    - neg, reciprocal
    - exp, log
    - sin, arcsin, cos, arccos, tan, arctan, sinh, arcsinh, cosh, arccosh, tanh, arctanh
    - gaussian, inv_gaussian
- binary ops
  - add, multiply
- sign

The nice thing about all these ops is that they all have inverses or pseudo inverses
- reversible
  - permute <-> permute 
    - 0 1 2 3 -permute(2 3 0 1)-> 2 3 0 1
    - 2 3 0 1 -permute(2 3 0 1)-> 0 1 2 3
  - group <-> ungroup 
    - 0 1 2 3 -group(a b (c d))-> 0 1 a
    -  0 1 a -ungroup(a b (c d))-> 0 1 2 3
  - squeeze <-> unsqueeze
    -  0 1 2 3 -squeeze(a 1 b c 1 d)-> 0 (1) 2 3 (1) 5
    -  0 (1) 2 3 (1) 5 -unsqueeze(a (1) b c (1) d)-> 0 1 2 3
 -  unary ops
 -  binary ops
- irreversible
  - slice <~> pad 
    - x -slice[0, 1:-1, 3:]-> x[0, 1:-1, 3:]
    - x[0, 1:-1, 3:] ~pad[0, 1:-1, 3:]~> x
  - reduce/select <~> repeat
    - 0 1 2 3 -reduce(a (b) c d | sum)-> 0 (1) 2 3
    - 0 (1) 2 3 ~repeat(a {b: 3} c d)~> 0 1 2 3
  - fold <~> unfold
    - `foldsize(dim_size, kernet_size, stride) = (dim_size - 1)*stride + 1`
    - `unfoldsize(dim_size, kernet_size, stride) = (dim_size - 1)/stride + 1`
    - 0 1 2 3 4 -fold(a {b: k s} c d | sum)-> 0 foldsize(1, 2, s)*k 2 3
    - 0 foldsize(1, 2, s)*k 2 3 ~unfold(a {b: k s} c d)~> 0 1 2 3 4
  - sign

## Example Derived ops

- Pow
  - Pow(A, B) = A^B = exp(log(A^B)) = exp(B*log(A))
- Stride
  - pad
  - ungroup
  - select
  - ungroup
- torch repeat
  - unsqueeze
  - repeat
  - group
- repeat interleave
  - unsqueeze
  - repeat
  - group
- Einsum
  - for each input
    - unsqueeze: align missing axes
    - repeat: fill in missing axes from other input
  - multiply
  - reduce | sum
  - optional squeeze
- Reshape
  - just group (easy) 
  - or just ungroup (easy)
  - or ungroup/group (NP hard)
  - or group/ungroup (fallback)
- Convolution
  - pad
  - unfold
  - stride #dialation
  - ungroup
  - multiply
  - reduce | sum
  - squeeze
- Transpose Convolution
  - unsqueeze
  - repeat interleave
  - multipy
  - group
  - pad interleave # undialation
  - fold
  - slice
- concat
  - input 1 before pad
  - input 2 after pad
  - add
- Stack
  - input 1 unsqueeze
  - input 2 unsqueeze
  - cat
- Activation functions
  - relu
    - (x + abs(x))/2
  - leakyrelu
    - (1 + slope)/2 * x + (1 - slope)/2 * abs(x)
  - softplus
    - log(1 + exp(beta * x))/beta
  - sigmoid
    - 1/(1+exp(-x))

## Controller Node

Takes a list of tensors in and a state dict.
returns a list of tensors and a state dict.

## Pipeline for inputs and outputs to/from IR

- from_
  - Cannonicalizer (1 pass)
    - From framework to IR
  - Refiner (n passes)
    - From IR to IR
- to_
  - Lowerer (m passes)
    - From IR to IR with Placeholders
  - Reifier (1 pass)
    - From IR with Placeholders to framework