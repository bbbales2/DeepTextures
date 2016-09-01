import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Misc import *
import numpy

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

        print x.shape

        print numpy.mean(x.flatten())

    dys, dxs = numpy.gradient(filtrd)
    dys = -dys

    angles, histogram = hog.build2dHist(dys, dxs)

    dxs = numpy.array(dxs_)
    dys = numpy.array(dys_)

    nans = numpy.where(numpy.logical_or(numpy.isnan(dxs_), numpy.isnan(dys_)))

    dxs[nans] = 0
    dys[nans] = 0

    magnitudes = numpy.sqrt(dxs**2 + dys**2 + 1e-10)
    rs = magnitudes

    if len(rs.shape) == 2:
        magnitudes[0, :] = 1e-12
        magnitudes[-1, :] = 1e-12
        magnitudes[:, 0] = 1e-12
        magnitudes[:, -1] = 1e-12

    magnitudes[nans] = 0

    phis = numpy.arctan2(dys, dxs)

    phis += (phis < 0) * 2.0 * numpy.pi / (1 + 1e-15)

    phis = phis.flatten()

    phiEdges = numpy.linspace(0.0, 2 * numpy.pi, N + 1)
    centers = list((phiEdges[:-1] + phiEdges[1:]) / 2.0)
    centers.append(centers[-1] + centers[0] + 1e-15)
    centers.insert(0, -centers[0])

    centers = numpy.array(centers)

    # This next bit of code I tried to use to do an interpolated bin
    # So you accumulate in each bin based on distances from the value you're binning to the bin centers
    #
    #print phiEdges
    #print phis[319777]

    #indices = numpy.searchsorted(centers, phis)

    #dl = phis - centers[indices - 1]
    #dr = centers[indices] - phis

    #factor = dr / (dl + dr)

    #leftPhis = phis - centers[0]
    #leftPhis += (leftPhis < 0) * 2.0 * numpy.pi / (1 + 1e-15)
    #rightPhis = numpy.array(phis)
    #rightPhis[numpy.where(rightPhis > centers[-2])] = 1e-15

    #print factor[0]

    magnitudes = magnitudes.flatten()

    #histogram1 = numpy.histogram(leftPhis, bins = phiEdges, weights = factor * magnitudes)[0]
    #histogram2 = numpy.histogram(rightPhis, bins = phiEdges, weights = (1.0 - factor) * magnitudes)[0]

    #histogram = histogram1 + histogram2

    histogram = numpy.histogram(phis, bins = phiEdges, weights = magnitudes)
    histogram = histogram[0]

        return [f_val, np.array(f_grad.ravel(), dtype=float)]

    result = minimize(f, init,
                          method='L-BFGS-B',
                          jac=True,
                          bounds=bounds,
                          callback=callback,
                          options=minimize_options)
    return result


