#%%

import os

os.chdir('/home/bbales2/DeepTextures')

import numpy
import bisect
import skimage.io
import scipy.ndimage
import matplotlib.pyplot as plt

#%%

im = skimage.io.imread('Images/2.png')

plt.imshow(im)
plt.show()

im = im.astype('float')[:, :, :3]

h = numpy.array(sorted(-numpy.linspace(-numpy.pi, numpy.pi, 40, endpoint = False)))
ht = numpy.power(numpy.sin(4 * h) + 1.0, 4.0)
plt.polar(h, ht)
plt.show()
ht /= sum(ht)
#%%

def histandderiv(im):
    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]

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

    dldh3 = (h3 - ht)

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

    return l, h3, numpy.rollaxis(numpy.array((dr, dg, db)), 0, 3)

l1, h1, dr1 = histandderiv(im)

plt.polar(h, h1)
plt.show()

im3 = im.copy()

ii = 71
jj = 11

im3[ii, jj, 0] *= 1.0001
#im3 *= 1.001

l2, h2, dr2 = histandderiv(im3)

plt.polar(h, h2)
plt.show()

#print l1, l2, -(l2 - l1) / (im[0, 0, 0] - im3[0, 0, 0]), dr1[0, 0], dr2[0, 0]

print (l2 - l1) / ((im[ii, jj, 0] - im3[ii, jj, 0])), dr1[ii, jj], dr2[ii, jj]#, dldmag1[1], dldmag2[1]

idxs = []
for i, a in enumerate(angle):
    idxs.append(bisect.bisect_left(h, a))
#%%
im0 = im.copy()
for i in range(200):

    l1, h1, dr1 = histandderiv(im0)

    im0 += 1000.0 * dr1 / numpy.linalg.norm(dr1.flatten())

    print l1
    #print h1
    plt.polar(h, h1)
    plt.show()

    im1 = im0.copy()
    im1 -= im1.flatten().min()
    im1 /= im1.flatten().max()

    plt.imshow(im1)
    plt.show()
#%%
a0 = numpy.random.rand(4)
at = numpy.random.rand(4)
#%%
a1 = numpy.copy(a0)
a1[0] *= 1.000001

a2 = a0 / sum(a0)#numpy.linalg.norm(a0)
a3 = a1 / sum(a1)#numpy.linalg.norm(a1)

a4 = 0.5 * sum((a2 - at)**2)
a5 = 0.5 * sum((a3 - at)**2)

print (a5 - a4) / (a1[0] - a0[0]), (a0 / sum(a0)) * (sum(a0) - a0) / (sum(a0)**2)
print (a3 - a2) / (a1[0] - a0[0]), (sum(a0) - a0) / (sum(a0)**2)
#%%
d = numpy.zeros((4, 4))

for i in range(4):
    for j in range(4):
        d[i, j] = -a0[i] / sum(a0)**2

for i in range(4):
    d[i, i] += 1 / sum(a0)

#tmp =

print numpy.dot(a0 / sum(a0) - at, -a0 / sum(a0)**2) + a0 / sum(a0)**2 - at / sum(a0)
print numpy.dot(a0 / sum(a0) - at, d)

#%%
print (a5 - a4) / (a1[0] - a0[0]), numpy.dot(a2, (numpy.sqrt(sum(a0**2)) - a0**2 / numpy.sqrt(sum(a0**2))) / sum(a0**2))
print (a3 - a2) / (a1[0] - a0[0]), (numpy.sqrt(sum(a0**2)) - a0**2 / numpy.sqrt(sum(a0**2))) / sum(a0**2)
#%%

im3 = numpy.zeros((256, 256))

Ax = numpy.array((im.shape[0] * im.shape[1], im.shape[0] * im.shape[1]))

im2[:-2] - im2[2:]

for i in range(256):
    for j in range(256):
        im3[i, j] = im2[i * 256 + j]

plt.imshow(im3)
plt.show()
#%%

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