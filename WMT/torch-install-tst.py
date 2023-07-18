import torch
x = torch.rand(5, 3)
print(x)

import torch
if torch.cuda.is_available():
    print("cuda OK")
else:
    print("cuda FAILED")

