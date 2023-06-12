import time
import torch
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

#############################################
# Get Device for Training
# -----------------------
# We want to be able to train our model on a hardware accelerator like the GPU or MPS,
# if available. Let's check to see if `torch.cuda <https://pytorch.org/docs/stable/notes/cuda.html>`_
# or `torch.backends.mps <https://pytorch.org/docs/stable/notes/mps.html>`_ are available, otherwise we use the CPU.

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


if 0:

    print(torch.__version__)

    # scalar
    scalar = torch.tensor(7)
    print(scalar)
    print(f"scalar.ndim = {scalar.ndim}\n")
    print(f"scalar.item = {scalar.item()}\n")
    
    # vector
    vector = torch.tensor([7,8])
    print(vector)
    print(f"vector.ndim = {vector.ndim}\n")
    #print(f"vector.item = {vector.item()}\n")
    
    # matrix
    MATRIX = torch.tensor([[7,8]])
    print(MATRIX)
    print(f"MATRIX.ndim = {MATRIX.ndim}\n")
    print(f"MATRIX.shape = {MATRIX.shape}\n")
    
    # tensor
    print("# tensor")
    TENSOR = torch.tensor([[[7,8],[1,2],[4,5]],
                           [[1,2],[3,4],[5,6]]
                           ])
    print(TENSOR)
    print(f"TENSOR.ndim = {TENSOR.ndim}\n")
    print(f"TENSOR.shape = {TENSOR.shape}\n")
    print(f"TENSOR[0] = {TENSOR[0]}\n")
    print(f"TENSOR[1] = {TENSOR[1]}\n")
    print(f"TENSOR[1][0] = {TENSOR[1][0]}\n")
    
    # tensor
    print("# tensor")
    TENSOR = torch.tensor([[[1,2,3],
                            [4,5,6],
                            [7,8,9]]])
    print(TENSOR)
    print(f"TENSOR.ndim = {TENSOR.ndim}\n")
    print(f"TENSOR.shape = {TENSOR.shape}\n")
    print(f"TENSOR[0] = {TENSOR[0]}\n")
    #print(f"TENSOR[1] = {TENSOR[1]}\n")
    print(f"TENSOR[0][0] = {TENSOR[0][0]}\n")
    
    # random tensors
    print("# create random tensor")
    rnd = torch.rand(3,4,5)
    print(rnd)
    print(f"rnd.ndim = {rnd.ndim}\n")
    print(f"rnd.shape = {rnd.shape}\n")
    print(f"rnd[0] = {rnd[0]}\n")
    #print(f"rnd[1] = {rnd[1]}\n")
    print(f"rnd[0][0] = {rnd[0][0]}\n")
    
    # random image
    print("# create random image tensor, h=224,w=224,color-channels=3 (R,G,B)")
    rnd_image = torch.rand(size=(224,224,3))
    print(rnd_image)
    print(f"rnd_image.ndim = {rnd_image.ndim}\n")
    print(f"rnd_image.shape = {rnd_image.shape}\n")
    #print(f"rnd_image[0] = {rnd_image[0]}\n")
    #print(f"rnd_image[1] = {rnd_image[1]}\n")
    #print(f"rnd_image[0][0] = {rnd_image[0][0]}\n")
    
    # see pytorch fundamentals notebook for more detail
    
    # zeroes and ones
    print("# zeroes and ones")
    zeroes = torch.zeros(size=(3,4))
    print(zeroes)
    print(f"zeroes.ndim = {zeroes.ndim}\n")
    print(f"zeroes.shape = {zeroes.shape}\n")
    
    print("# zeroes and ones")
    ones = torch.ones(size=(2,2))
    print(ones)
    print(f"ones.ndim = {ones.ndim}\n")
    print(f"ones.shape = {ones.shape}\n")
    print(f"ones.dtype = {ones.dtype}\n")
    
    print("# Creating a range of tensors and tensors-like")
    tr = torch.arange(0,10)
    print(tr)
    print(f"tr.ndim = {tr.ndim}\n")
    print(f"tr.shape = {tr.shape}\n")
    print(f"tr.dtype = {tr.dtype}\n")
    
    print("# Creating tensors-like")
    tll = torch.ones_like(input=tr)
    print(tll)
    print(f"tll.ndim = {tll.ndim}\n")
    print(f"tll.shape = {tll.shape}\n")
    print(f"tll.dtype = {tll.dtype}\n")
    
    print("# vary dtype")
    ttdd = torch.ones(size=(1,10),dtype=torch.float32,
                      device="cpu",
                      requires_grad=False)
    print(ttdd)
    print(f"ttdd.ndim = {ttdd.ndim}\n")
    print(f"ttdd.shape = {ttdd.shape}\n")
    print(f"ttdd.dtype = {ttdd.dtype}\n")
    
    ttdd_f16 = ttdd.type(torch.float16)
    print(ttdd_f16)
    
    print(f"ttdd.dtype = {ttdd.dtype}\n")
    print(f"ttdd.shape = {ttdd.shape}\n")
    print(f"ttdd.device = {ttdd.device}\n")
    
    # dot product - Note: stays on same device
    # Note: innter dimensions must match
    # Note: resultant will be dimensions of outer dimensions
    
    print("# Dot product")
    m1 = torch.tensor([[1,2],[3,4]],
                      dtype=torch.float32,
                      device="cpu",
                      requires_grad=False)
    
    start = time.time()
    m2 = m1
    for i in range(1,100):
        m2 = m2.matmul(m1)
        m1 = m2
        end = time.time()
        print(f"Execution time = {end-start}\n")
        
        print(m2)
        print(f"m2.dtype = {m2.dtype}\n")
        print(f"m2.shape = {m2.shape}\n")
        print(f"m2.device = {m2.device}\n")
        
        # various tensor operations
        
        # Reshaping
        # View
        # Stacking
        # Squeeze
        # Unsqueeze
        # Permute
        
        x = torch.arange(1.,10.)
        x_reshaped = x.reshape(1,9)
        print(x)
        print(x_reshaped)
        z = x.view(9,1)
        print(z)

        # Stack Tensors
        print("# play with stacking")
        z_stacked = torch.stack([x,x,x,x])
        print(z_stacked)
        z_stacked = torch.stack([x,x,x,x],dim=1)
        print(z_stacked)

        # Squeese, unsqueese
        print("# Squeese, unsqueese")
        z = torch.zeros(2,1,2,1,2)
        y = torch.squeeze(z)
        print(f"z.size() = {z.size()}\ny.size() = {y.size()}\n")

        # Permute
        # Create a tensor
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Permute the tensor
        t_permuted = torch.permute(t, (1, 0))

        # Print the tensor
        print(t_permuted)

        print("# Permute")        
        x = torch.randn(2,3,5)
        print(f"x = {x}, x.size = {x.size()}\n")
        y = torch.permute(x,(2,1,0))
        print(f"y = {y}, y.size = {y.size()}\n")

        # Next
        z = np.array([1,2,3])
        print(f"z = {z}, type(z) = {type(z)}\n")
        z1 = torch.from_numpy(z)
        print(f"z1 = {z1}, type(z1) = {type(z1)}\n")
        z2 = torch.Tensor.numpy(z1)
        print(f"z2 = {z2}, type(z2) = {type(z2)}\n")

# not quite random
print("# Not quite random if use manual_seed")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
r_A = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED)
r_B = torch.rand(3,4)
print(r_A==r_B)
