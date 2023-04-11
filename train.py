import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from TransFormer import TransFormer
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.randint(0, 30, (300,), dtype=torch.float)
    Y = X.pow(2)
    print("X", X)
    print("Y", Y)
    model = TransFormer()
    if device == 'cuda':
        X = X.to(device)
        model.to(device)
    print(model)

    optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.005)
    loss_function = nn.MSELoss()

    epochs = 50000

    X = torch.Tensor(X.reshape(300, 1))
    Y = torch.Tensor(Y.reshape(300, 1))
    
    epoch = 0
    while True:
        epoch += 1
        
        prediction = model(X)
        loss = loss_function(prediction, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch : {epoch:d} Loss : {loss.data.numpy():.4f}")
        
        if loss.data.numpy() < 1:
            print(f"Epoch : {epoch:d} Loss : {loss.data.numpy():.4f}")
            break
        

    # for epoch in range(epochs):
    #     if epoch % 500 == 0:
    #         prediction = model(X)
    #         loss = loss_function(prediction, Y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         print("Loss : %.4f" % loss.data.numpy())
    
    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("\nOptimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    
    torch.save(model.state_dict(), "./model/model.pt")