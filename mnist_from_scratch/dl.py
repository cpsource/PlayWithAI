# where.from - https://www.youtube.com/watch?v=c36lUUr864M

import torch

x = torch.randn(3,requires_grad=True)
y = x + 2

print(x)
print(y)

z = y*y*2
#z = z.mean()
print(z)
# dz/dx
q = torch.tensor([1.0,2.0,3.0])
z.backward(q)
print(x.grad)

weights = torch.ones(3,requires_grad=True)

print(f"weights.grad = {weights.grad}")

for epoch in range(1):
    model_output = (weights*2).sum()
    print(f"model_output = {model_output}")
    model_output.backward()
    print(f"weights.grad = {weights.grad}")
    weights.grad.zero_()

