import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def __call__(self, x):
        return self.linear(x)

model = MyModel()

x = torch.randn(10, 10)
print(f"x: {x}\n")

logits = model(x)
print(f"logits: {logits}\n")

prediction = logits.argmax(dim=1)

print(f"prediction: {prediction}\n")
