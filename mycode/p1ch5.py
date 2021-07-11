import torch
import matplotlib.pyplot as plt
import torch.optim as optim


class model():

    def __init__(self, w = None, b = None):
        self.w = w
        self.b = b

    def out(self, t_u):
        return self.w*t_u + self.b

def model(t_u, w, b):
    return w * t_u + b

def dmodel_dw(t_u, w, b):
    return t_u

def dmodel_db(t_u, w, b):
    return 1.0

def loss_fn(t_p, t_c):
    squared_diff = (t_p - t_c)**2
    return squared_diff.mean()

def dloss_fn(t_p, t_c):
    return 2 * (t_p - t_c) / t_p.size(0)

def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

def print_loss_and_para(loss, w, b, message = None):
    if message != None:
        print('= ', message)
    else:
        print('========================')
    
    print('|| loss : ', loss)
    print('|| w    : ', w)
    print('|| b    : ', b)
    print('========================')

def training_loop(n_epochs, t_u, t_c, learning_rate, 
                params, print_params = True):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        params = params - learning_rate * grad
        if epoch % 10 == 0 or epoch == n_epochs:
            print('Epoch %d, loss: %f' % (epoch, loss))
            if print_params:
                print('     weights : ', params[0])
                print('     bias    : ', params[1])

    return params

def training_loop_autograd(n_epochs, t_u, t_c, learning_rate, 
                params, print_params = True):
    for epoch in range(1, n_epochs + 1):

        if params.grad is not None:
            params.grad.zero_()

        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        loss.bachward()

        with torch.no_grad():
            params = params - learning_rate * grad
        
        if epoch % 10 == 0 or epoch == n_epochs:
            print('Epoch %d, loss: %f' % (epoch, loss))
            if print_params:
                print('     weights : ', params[0])
                print('     bias    : ', params[1])

    return params


def traning_loop_autograd_optim(n_epochs, optimizer, t_u, t_c, 
                                params, print_params = True):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            pint('Epoch %d, Loss %f' % (epoch, float(loss)))
    
    return params
# gradient descent approximately step of w and b is 0.1
def approximately_devirative():
    global w, b, t_c
    delta = 0.1
    learning_rate = 1e-2
    print_loss_and_para(loss_fn(model(t_u, w, b), t_c), w, b)

    loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c)
        - loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)

    loss_rate_of_change_b = (loss_fn(model(t_u, w, b + delta), t_c)
        - loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)

    w = w - learning_rate * loss_rate_of_change_w
    b = b - learning_rate * loss_rate_of_change_b
    print_loss_and_para(loss_fn(model(t_u, w, b), t_c), w, b)

if __name__ == '__main__':

    # our data
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
    t_u = torch.tensor(t_u)
    t_c = torch.tensor(t_c)

    # param
    w = torch.ones(())
    b = torch.zeros(())

    # normalizing
    t_un = 0.1 * t_u

    params = training_loop(
        n_epochs = 5000,
        t_u = t_un,
        t_c = t_c,
        learning_rate = 1e-2,
        params = torch.tensor([1.0, 0.0])
    )


    # visualizing the data
    t_p = model(t_un, *params)
    # fig = plt.figure(dpi = 600)
    plt.xlabel("*F")
    plt.ylabel('*C')
    plt.plot(t_u.numpy(), t_p.detach().numpy())
    plt.plot(t_u.numpy(), t_c.numpy(), 'o')
    plt.plot(1,2,'ro')
    plt.show()

    # split the dataset
    n_saples = t_u.shape[0]
    n_vmal = int(0.2 * n_samples)
    shuffed_indieces = torch.randperm(n_samples)
    train_indices = shuffed_indieces[:-n_val]
    val_indices = shuffed_indieces[-n_val:]