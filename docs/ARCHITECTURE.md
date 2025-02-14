# Architecture

> [!WARNING]
> This is an early work in progress.

## Overview

- It is hard to automatically determine things about models, such as total_flops, memory_reads, and memory_writes
- It is hard to remove batch dimensions from models when needed
- Most IRs are made for performance versus portability

The vast majority of neural networks are just Tensor Directed Acyclic Graphs (TDAGs) with 
control flow (conditional branching, dynamic-length loops, dynamic shapes...) only at the top level.

## Goal

- Split the model into a series of TDAGs
- TDAGs have very few core operations
- Each TDAG can be thought of as a "jit.trace" of a neural network
- TDAGs' input is a list of tensors
- TDAGs output a list of tensors
- Connect TDAGs via control flow only at the top level
- Enable developers to extract TDAGs for reuse in other applications easily
- Enable easy graph modification

## Implementation

### Tensor Types

- input: defined at runtime
- parameter: learnable parameter
- buffer: a non-learnable parameter that can change at runtime
- const: a non-learnable parameter that cannot change at runtime

### Op IR

- Each Op has inputs and a spec
- Inputs contain names of ops and tensors in the graph
- Einops are nice representations of tensor operations that generalize well to different platforms and frameworks
  - Further reduce them to their simplest form
  - Can be used to construct many other ops

#### Deinop Philosophy

The thing about all these ops is that they have a nice symmetry:
- reversible
  - permute <-> permute 
    - 0 1 2 3 -permute(2 3 0 1)-> 2 3 0 1
    - 2 3 0 1 -permute(2 3 0 1)-> 0 1 2 3
  - group <-> ungroup 
    - 0 1 2 3 -group(a b (c d))-> 0 1 a
    - 0 1 a -ungroup(a b (_ 3))-> 0 1 2 3
  - squeeze <-> unsqueeze
    - 0 1 2 3 -squeeze(a 1 b c 1 d)-> 0 (1) 2 3 (1) 5
    - 0 (1) 2 3 (1) 5 -unsqueeze(a (1) b c (1) d)-> 0 1 2 3
- irreversible
  - slice <~> pad 
    - x -slice[0, 1:-1, 3:]-> x[0, 1:-1, 3:]
    - x[0, 1:-1, 3:] ~pad[0, 1:-1, 3:]~> x
  - reduce/select <~> repeat
    - 0 1 2 3 -reduce(a (b) c d | sum)-> 0 (1) 2 3
    - 0 (1) 2 3 ~repeat(a {b: 3} c d)~> 0 1 2 3
  - fold <~> unfold
    - `foldsize(dim_size, kernel_size, stride) = (dim_size//kernel_size - 1)*stride + kernel_size`
    - `unfoldsize(dim_size, kernel_size, stride) = ((dim_size - kernel_size)/stride + 1)*kernel_size`
    - 0 1 2 3 -fold(a {b: k s} c d | sum)-> 0 foldsize(1, k, s) 2 3
    - 0 foldsize(1, k, s) 2 3 ~unfold(a {b: k s} c d)~> 0 1 2 3

#### Deconstructed Einops (Deinops)

- permute
  - permutes dimensions
  - "b c h w -> b h w c"
  - `torch.permute(x, [3,2,1,0])`
- group
  - basically reshapes each dimension by multiplying 2 dimensions together
  - $a b c d -> a b (c d)$ 
  - `torch.reshape(x, (a, b, c * d))`
- ungroup
  - basically reshapes each dimension by factoring out dimensions, can have one wildcard 
  - $n c (img, 10) -> n c img 10$ 
  - `torch.reshape(n, c, -1, 10)`
- squeeze
  - helper function for group, removes a dimension where size equals one
  - $a b 1 c d 1 -> a b c d$ 
  - `torch.squeeze(x, dim=(2, 5))`
- unsqueeze
  - helper function for ungroup, inserts a dimension where size equals one
  - $a b c d -> a b 1 c d 1$ 
  - `torch.unsqueeze(x, dim=(2, 5))`
- slice
  - same as numpy without step (except I prefer inclusive indices for slicing)
  - `x[..., 1:, :, 1:, :1, 1:2, -1:, :-1]`
- pad
  - pads the input
  - a b -> 1 <- a -> 1 2 <- c -> 0
  - `torch.nn.functional.pad(x, (1, 1, 2, 0))`
- reduce
  - performs an operation across the entire axis that reduces it to a size of s
  - $n {c: 1} h w -> n 1 h w | mean$
  - `torch.sum(x, dim=(1,3), keep_dims=True)`
- select
  - helper function for reduce to an index
  - $n {c: 1} h w -> n 1 h w$
  - `torch.unsqueeze(x[:, 1, :, :], 1)`
- repeat
  - performs an operation that repeats the dim of size 1
  - $(1) (1) h w -> (3) (10) h w$
- fold
  - n c h*a w*b -> n c {h: a 3} {w: b 3}
  - `x = F.fold(x, output_size=(h, w), kernel_size=(a, b), stride=3, dilation=1, padding=0)`
- unfold
  - n c {h: a 3} {w: b 3} -> n c h*a w*b
  - sort of `x = F.unfold(pad(x), kernel_size=(a, b), stride=3, dilation=1, padding=0)`

#### Other Tensor Ops
- unary ops
    - neg, reciprocal
    - exp, log
    - sin, arcsin, cos, arccos, tan, arctan, sinh, arcsinh, cosh, arccosh, tanh, arctanh
    - gaussian, inv_gaussian
- binary ops
  - add, multiply
- sign

#### Example Derived Ops

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
- Einsum (matmul, batched matmul)
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
  - stride #dilation
  - ungroup
  - multiply
  - reduce | sum
  - squeeze
- Transpose Convolution
  - unsqueeze
  - repeat interleave
  - multiply
  - group
  - ungroup pad group #undilation
  - fold
  - slice
- concat
  - input 1 before pad
  - input 2 after pad
  - add
- Stack
  - input 1 unsqueeze
  - input 2 unsqueeze
  - concat
- Activation functions
  - relu
    - (x + abs(x))/2
  - leakyrelu
    - (1 + slope)/2 * x + (1 - slope)/2 * abs(x)
  - softplus
    - log(1 + exp(beta * x))/beta
  - sigmoid
    - 1/(1+exp(-x))

### Controller Node

- Takes a list of tensors in and a state dict
- Returns a list of tensors and a state dict

### Pipeline for Inputs and Outputs to/from IR

- from_
  - Canonicalizer (1 pass)
    - From framework to IR
  - Refiner (n passes)
    - From IR to IR
- to_
  - Lowerer (m passes)
    - From IR to IR with Placeholders
  - Reifier (1 pass)
    - From IR with Placeholders to framework
