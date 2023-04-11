import torch
from TransFormer import TransFormer

if __name__ == "__main__":
    device = torch.device('cpu')
    model = TransFormer()
    model.load_state_dict(torch.load('./model.pt', map_location=device))

    x = torch.randint(1, 30, (1,), dtype=torch.float)
    print(x)
    y = model(x)
    print("x", x)
    print("y", y)