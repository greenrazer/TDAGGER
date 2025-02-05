import torch

def run_reshape_tests():

    x = torch.rand((5,4,6,64,32,12,3,4))

    new_shapes = [
        (5,4,6,64,32,36,4),
        (5,4,6,64,32,144),
        (120,64,32,12,12),
        (120,2048,12,12),
        (120,2048,144),
        (144,2048,120),
        (2048,120, 144),
        (6, 4, 5, 64, 4, 12, 32, 3),
        (6, 4, 5, 256, 12, 96),
        (6, 256, 12, 96, 4, 5),
        (256, 72, 96, 20),
    ]
    for s in new_shapes:
        ref = x.reshape(s)
        test = x.reshape(-1).reshape(s)
        assert torch.equal(ref, test)

if __name__ == "__main__":
    run_reshape_tests()