import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Misc import *
import numpy
import bisect

h = numpy.array(sorted(-numpy.linspace(-numpy.pi, numpy.pi, 40, endpoint = False)))
ht = numpy.power(numpy.sin(4 * h) + 1.0, 4.0)
plt.polar(h, ht)
plt.show()
ht /= sum(ht)

def histandderiv(im):
    r = im[0, 0, :, :]
    g = im[0, 1, :, :]
    b = im[0, 2, :, :]

    im3 = numpy.sqrt(r**2 + g**2 + b**2)

    im2 = scipy.ndimage.filters.gaussian_filter(im3, 0.5)

    dim2dr = r / im2
    dim2dg = g / im2
    dim2db = b / im2

    dx = numpy.pad(im2[:, : -2] - im2[:, 2:], ((0, 0), (1, 1)), mode = 'constant')
    dy = numpy.pad(im2[:-2, : ] - im2[2:, :], ((1, 1), (0, 0)), mode = 'constant')

    mag = numpy.sqrt(dx**2 + dy**2 + 1e-10)
    angle = numpy.arctan2(dy, dx).flatten()

    magf = mag.flatten()

    h2 = numpy.zeros(40)

    for i, a in enumerate(angle):
        h2[bisect.bisect_left(h, a)] += magf[i]

    total = sum(h2)

    h3 = h2 / total

    l = 0.5 * sum((h3 - ht)**2)

    #dldh3 = (h3 - ht)

    a0 = h2
    at = ht
    dldh2 = numpy.dot(a0 / sum(a0) - at, -a0 / sum(a0)**2) + a0 / sum(a0)**2 - at / sum(a0)

    dldmag = numpy.zeros(256 * 256)

    for i, a in enumerate(angle):
        dldmag[i] = dldh2[bisect.bisect_left(h, a)]

    dmagdx = dx / mag
    dmagdy = dy / mag

    tmp1 = dldmag.reshape((256, 256)) * dmagdx
    dldim3 = numpy.pad(tmp1[:, : -2] - tmp1[:, 2:], ((0, 0), (1, 1)), mode = 'constant')
    tmp2 = dldmag.reshape((256, 256)) * dmagdy
    dldim3 += numpy.pad(tmp2[:-2, : ] - tmp2[2:, :], ((1, 1), (0, 0)), mode = 'constant')

    dldim2 = scipy.ndimage.filters.gaussian_filter(dldim3, 0.5)

    dr = dldim2 * dim2dr
    dg = dldim2 * dim2dg
    db = dldim2 * dim2db

    return l, h3, numpy.array((dr, dg, db)).reshape((1, 3, r.shape[0], r.shape[1]))

def ImageSyn(net, constraints, init=None, bounds=None, callback=None, minimize_options=None, gradient_free_region=None):
    '''
    This function generates the image by performing gradient descent on the pixels to match the constraints.

    :param net: caffe.Classifier object that defines the network used to generate the image
    :param constraints: dictionary object that contains the constraints on each layer used for the image generation
    :param init: the initial image to start the gradient descent from. Defaults to gaussian white noise
    :param bounds: the optimisation bounds passed to the optimiser
    :param callback: the callback function passed to the optimiser
    :param minimize_options: the options passed to the optimiser
    :param gradient_free_region: a binary mask that defines all pixels that should be ignored in the in the gradient descent
    :return: result object from the L-BFGS optimisation
    '''

    if init==None:
        init = np.random.randn(*net.blobs['data'].data.shape)

     #get indices for gradient
    layers, indices = get_indices(net, constraints)

    #function to minimise
    def f(x):
        x = x.reshape(*net.blobs['data'].data.shape)
        net.forward(data=x, end=layers[min(len(layers)-1, indices[0]+1)])
        f_val = 0
        #clear gradient in all layers
        for index in indices:
            net.blobs[layers[index]].diff[...] = np.zeros_like(net.blobs[layers[index]].diff)

        for i,index in enumerate(indices):
            layer = layers[index]
            for l,loss_function in enumerate(constraints[layer].loss_functions):
                constraints[layer].parameter_lists[l].update({'activations': net.blobs[layer].data.copy()})
                val, grad = loss_function(**constraints[layer].parameter_lists[l])
                f_val += val
                net.blobs[layer].diff[:] += grad
            #gradient wrt inactive units is 0
            net.blobs[layer].diff[(net.blobs[layer].data == 0)] = 0.
            if index == indices[-1]:
                f_grad = net.backward(start=layer)['data'].copy()
            else:
                net.backward(start=layer, end=layers[indices[i+1]])

        if gradient_free_region!=None:
            f_grad[gradient_free_region==1] = 0

        log, h1, deriv = histandderiv(x)

        a = 2e7

        print 'loss: ', f_val
        print 'angle loss: ', a * log

        print 'ratio (image / angle loss): ', f_val / (a * log)

        plt.polar(h, h1)
        plt.show()

        return [f_val + a * log, np.array(f_grad.ravel() - a * deriv.ravel(), dtype=float)]

    result = minimize(f, init,
                          method='L-BFGS-B',
                          jac=True,
                          bounds=bounds,
                          callback=callback,
                          options=minimize_options)
    return result


