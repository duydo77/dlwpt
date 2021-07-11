import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from collections import OrderedDict

def training_loop(n_epochs, model, loss_fn, optimizer,
                    t_u_train, t_c_train, t_u_val, t_c_val):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(t_u_train)
        train_loss = loss_fn(train_t_p, t_c_train)

        with torch.no_grad():
            val_t_p = model(t_u_val)
            val_loss = loss_fn(val_t_p, t_c_val)
            assert val_loss.requires_grad == False
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 100 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f}," +
                    f" Validation loss {val_loss.item():.4f}")
            print('   params', list(model.parameters()))
            print()

def training_loop_NN():
    pass

def linear_module():

    # data
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
    t_u = torch.tensor(t_u)
    t_c = torch.tensor(t_c)
    t_un = 0.1 * t_u
    # train/val plit
    n_samples = t_u.shape[0]
    n_vals = int(0.2 * n_samples)

    shuffed_indieces = torch.randperm(n_samples)
    train_indieces = shuffed_indieces[0:-n_vals]
    val_indieces = shuffed_indieces[-n_vals:]

    t_u_train   = t_un[train_indieces].unsqueeze(1)
    t_u_val     = t_un[val_indieces].unsqueeze(1)
    t_c_train   = t_c[train_indieces].unsqueeze(1)
    t_c_val     = t_c[val_indieces].unsqueeze(1)

    # train
    linear_model = nn.Linear(1, 1)
    optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

    training_loop(
        n_epochs = 3000, 
        optimizer = optimizer,
        model = linear_model,
        loss_fn = nn.MSELoss(), 
        t_u_train = t_u_train,
        t_u_val = t_u_val, 
        t_c_train = t_c_train,
        t_c_val = t_c_val)

    print()
    print(linear_model.weight)
    print(linear_model.bias)

def neural():
    # data
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
    t_u = torch.tensor(t_u).unsqueeze(1)
    t_c = torch.tensor(t_c).unsqueeze(1)
    t_un = 0.1 * t_u
    # train/val plit
    n_samples   = t_u.shape[0]
    n_vals      = int(0.2 * n_samples)

    shuffed_indieces    = torch.randperm(n_samples)
    train_indieces      = shuffed_indieces[0:-n_vals]
    val_indieces        = shuffed_indieces[-n_vals:]

    t_u_train   = t_un[train_indieces]
    t_u_val     = t_un[val_indieces]
    t_c_train   = t_c[train_indieces]
    t_c_val     = t_c[val_indieces]

    seq_model = nn.Sequential(OrderedDict([
        ('hidden_linear', nn.Linear(1, 8)),
        ('hidden_activation', nn.Tanh()),
        ('output_linear', nn.Linear(8, 1))
    ]))

    optimizer = optim.SGD(seq_model.parameters(), lr = 1e-3)

    training_loop(
        n_epochs = 3000, 
        optimizer = optimizer,
        model = seq_model,
        loss_fn = nn.MSELoss(), 
        t_u_train = t_u_train,
        t_u_val = t_u_val, 
        t_c_train = t_c_train,
        t_c_val = t_c_val)
    
    print()
    print('output', seq_model(t_u_val))
    print('answer', t_c_val)
    print('hidden', seq_model.hidden_linear.weight.grad)

    # plot
    t_range = torch.arange(20., 90.)
    
    print(t_range.shape)

    plt.xlabel("*F")
    plt.ylabel("*C")
    plt.plot(t_u.numpy(), t_c.numpy(), 'o')
    plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
    plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
    plt.show()


if __name__ == '__main__':
    # linear_module()
    # neural()
    plt.plot([1,2], [3, 4])
    plt.show()