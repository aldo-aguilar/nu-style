# test functions
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter1d


def get_gradient_norm(model):
    norms = 0
    for param in model.parameters():
        norms += param.grad.data.detach().norm(2)
    return norms


def visualize_history_norm(history, history_name, zoom_axis=[], sigma=5):
    plt.plot(gaussian_filter1d(history, sigma))
    plt.xlabel('Iteration')
    plt.ylabel(history_name)
    if zoom_axis:
        plt.axis(zoom_axis)
    plt.show()


# this function should take a batch from the dataloader and:
#
# pass the batch through the model all at once and collect
# that output in a variable
#
# pass each item in the batch through the model by itself
# and collect the output such that it's the same shape as the
# original batch
#
# compare these two using torch.allclose to make sure they are
# the same!
def test_forward_pass(model, dataloader):
    data, _ = next(iter(dataloader))
    output_batch = model(data.float())
    output_single = []
    for datum in data:
        output_single.append(model(datum.float()))

    assert torch.allclose(output_batch, torch.cat(output_single)), 'Forward pass is batch dependent'


# this function should take a batch from the dataloader and:
#
# pass the batch through the model all at once then do a
# backwards and collect the gradient
#
# pass each item in the batch through the model by itself
# do backwards on each item (accummulating the gradient),
# and collect the gradient at the end
#
# compare these two using torch.allclose to make sure they are
# the same!
def test_backward_pass(model, dataloader, loss):
    model.zero_grad()
    data, targets = next(iter(dataloader))
    _loss_batch = loss(model(data.float()), targets.long())
    _loss_batch.backward()
    accumulated_batch = get_gradient_norm(model)

    model.zero_grad()
    for datum, target in zip(data, targets):
        _loss_single = loss(model(datum.float()), target.long().reshape(1))
        _loss_single.backward()
    accumulated_single = get_gradient_norm(model) / data.shape[0]

    assert torch.allclose(accumulated_batch.reshape(1), accumulated_single.reshape(1),
                          atol=1e-3), 'loss function is cross-linking data'


def test_gradient_flow(model, dataloader, loss, magnitude=-5, compare_prev_layers=True,
                       compare_prev_layers_magnitude=3):
    # pass data through the model, then compare the gradient at each
    # layer in the model. the gradient should never become really
    # tiny, as this means the earlier layers of the model will be
    # tough to train. your network is probably too deep!

    model.zero_grad()
    data, targets = next(iter(dataloader))
    _loss_batch = loss(model(data.float()), targets.long())
    _loss_batch.backward()
    grad_norms = []
    for param in model.parameters():
        grad_norms.append(param.grad.data.norm(2).detach())
    last_norm = grad_norms[-1]
    for norm in reversed(grad_norms[:(len(grad_norms) - 1)]):
        if comparelayers:
            assert not torch.log10(last_norm) - torch.log10(
                norm) > compare_prev_layers_magnitude, 'Early gradients vanish too quickly compared to later layers'
        assert torch.log10(last_norm) > magnitude, 'Tensor magnitude is too small'
        last_norm = norm