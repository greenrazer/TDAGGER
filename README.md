# SafeGraph

> [!WARNING]
> This is a work in progress

## Problem

- Neural Network formats are all quite difficult to translate between (.pt, .tflite, .onnx, ...)
- It is hard to tell things about models automatically, like total_flops, memory_reads, memory_writes

## Goal

- Split the model into a series of Tensor Directed Acyclic Graphs(TDAGs)(no conditional branching, dynamic-length loops, ...)
- TDAGs have very few core operations
- Each DDAG can be thought of as a "jit.trace" of a neural network.
- Connect DDAGs via control flow only at the top level (conditional branching, dynamic-length loops, ...)
- The top level can be thought of as a list of different traces each gated by a giant switch/match statement that decides whether 
  to pass the ouput into the switch/match statement again or to ouptut it 
- 

## Plan

- Core Ops
  - permute
    - permute dimensions
    - "b c h w -> b h w c"
    - `torch.permute(x, [3,2,1,0])`
  - slice
    - same as numpy (except I prefer inclusive indices for my slicing)
    - `x[..., 0, :, 1:, :1, 1:2, -1:, :-1:2]`
  - pad
    - pad the input
    - a b -> 1 <- a -> 1 2 <- c -> 0
    - `torch.nn.functional.pad(x, (1, 1, 2, 0))`
  - reduce
    - do an operation across the entire axis that reduces it to a size of 1
    - $n c h w -> n 1 h w | mean$
    - `torch.sum(x, dim=(1,3), keep_dims=True)`
  - repeat
    - $n 1 h w -> n c h w$
  - group
    - basically reshape each dimension by multiplying 2 dimensions together
    - $a b c d - > a b (c d)$ 
    - `torch.reshape(x, (a, b, c * d))`
  - ungroup
    - basically reshape each dimension by factoring out dimensions, can have one wild card 
    - $n c (img, 10) - > n c img 10$ 
    - `torch.reshape(n, c, -1, 10)`
  - squeeze
    - insert a dimension where size equals one
    - $ a b 1 c d 1 - > a b c d$ 
    - `torch.squeeze(x, dim=(2, 5))`
  - unsqueeze
    - insert a dimension where size equals one
    - $ a b c d - > a b 1 c d 1$ 
    - `torch.unsqueeze(x, dim=(2, 5))`
  - fold (or tile)
    - n c h a w b -> n c {2 <- h -> 2: a 3 2} {2 <- w -> 2: b 3 2}
    - sort of `x = F.fold(x, output_size=(h, w), kernel_size=(a, b), stride=3, dilation=2, padding=0)`
  - unfold (or tile)
    - n c {2 <- h -> 2: a 3 2} {2 <- w -> 2: b 3 2} -> n c h a w b
    - sort of `x = F.unfold(pad(x), kernel_size=(a, b), stride=3, dilation=2, padding=2)`
  - 
