# test the matrices for the interpolation, just numpy for now.

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/ianczekala/Documents/ALMA/million-points-of-light")

# convert from arcseconds to radians
arcsec = np.pi / (180.0 * 3600) # [radians]  = 1/206265 radian/arcsec


def sky_plane(alpha, dec, a=1, delta_alpha=0.0, delta_delta=0.0, sigma_alpha=1.0*arcsec,
              sigma_delta=1.0*arcsec, Omega=0.0):
    '''
    alpha: ra (in radians)
    delta: dec (in radians)
    a : amplitude
    delta_alpha : offset (in radians)
    delta_dec : offset (in radians)
    sigma_alpha : width (in radians)
    sigma_dec : width (in radians)
    Omega : position angle of ascending node (in degrees east of north)
    '''

    return a * np.exp(-( (alpha - delta_alpha)**2/(2 * sigma_alpha**2) + \
                        (dec - delta_delta)**2/(2 * sigma_delta**2)))


def fourier_plane(u, v, a=1, delta_alpha=0.0, delta_delta=0.0, sigma_alpha=1.0*arcsec,
              sigma_delta=1.0*arcsec, Omega=0.0):
    '''
    Calculate the Fourier transform of the Gaussian.
    '''

    return 2 * np.pi * a * sigma_alpha * sigma_delta * np.exp(- 2 * np.pi**2 * (sigma_alpha**2 * u**2 + sigma_delta**2 * v**2) - 2 * np.pi * j * (delta_alpha * u + delta_delta * v))


def fftspace(width, N):
    '''Oftentimes it is necessary to get a symmetric coordinate array that spans ``N``
     elements from `-width` to `+width`, but makes sure that the middle point lands
     on ``0``. The indices go from ``0`` to ``N -1.``
     `linspace` returns  the end points inclusive, wheras we want to leave out the
     right endpoint, because we are sampling the function in a cyclic manner.'''

    assert N % 2 == 0, "N must be even."

    dx = width * 2.0 / N
    xx = np.empty(N, np.float)
    for i in range(N):
        xx[i] = -width + i * dx

    return xx


# Let's plot this up and see what it looks like
N_alpha = 128
N_dec = 128
img_radius = 15.0 * arcsec

# full span of the image
ra = fftspace(img_radius, N_alpha) # [arcsec]
dec = fftspace(img_radius, N_dec) # [arcsec]

# fill out an image
img = np.empty((N_dec, N_alpha), np.float)

for i,delta in enumerate(dec):
    for j,alpha in enumerate(ra):
        img[i,j] = sky_plane(alpha, delta)

fig,ax = plt.subplots(nrows=1)
ax.imshow(img, origin="upper", interpolation="none", aspect="equal")
ax.set_xlabel(r"$\Delta \alpha \cos \delta$")
ax.set_ylabel(r"$\Delta \delta$")
ax.set_title("Input image")
fig.savefig("input.png", dpi=300)

# pre-multiply the image by the correction function

# calculate the Fourier coordinates
dalpha = (2 * img_radius)/N_alpha
ddelta = (2 * img_radius)/N_dec

us = np.fft.rfftfreq(N_alpha, d=dalpha)
vs = np.fft.fftfreq(N_dec, d=ddelta)


# calculate the FFT, but first shift all axes.
# normalize output properly
vis = dalpha * ddelta * np.fft.rfftn(np.fft.fftshift(img), axes=(0,1))



# calculate the corresponding u and v axes
XX, YY = np.meshgrid(us, vs)

# left, right, bottom, top
vs_limit = np.fft.fftshift(vs)

ext = [us[0], us[-1], vs_limit[-1], vs_limit[0]]

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(7, 5))
ax[0,0].set_title("numerical")
ax[0,0].imshow(np.real(np.fft.fftshift(vis, axes=0)), origin="upper", interpolation="none", aspect="equal", extent=ext)
ax[1,0].imshow(np.imag(np.fft.fftshift(vis, axes=0)), origin="upper", interpolation="none", aspect="equal", extent=ext)

ax[0,1].set_title("analytical")
vis_analytical = fourier_plane(XX, YY)
ax[0,1].imshow(np.real(np.fft.fftshift(vis_analytical, axes=0)), origin="upper", interpolation="none", aspect="equal", extent=ext)
ax[1,1].imshow(np.imag(np.fft.fftshift(vis_analytical, axes=0)), origin="upper", interpolation="none", aspect="equal", extent=ext)

# compare to the analytic version
ax[0,2].set_title("difference")
im_real = ax[0,2].imshow(np.real(np.fft.fftshift(vis - vis_analytical, axes=0)), origin="upper", interpolation="none", aspect="equal", extent=ext)
plt.colorbar(im_real, ax=ax[0,2])
im_imag = ax[1,2].imshow(np.imag(np.fft.fftshift(vis - vis_analytical, axes=0)), origin="upper", interpolation="none", aspect="equal", extent=ext)
plt.colorbar(im_imag, ax=ax[1,2])

fig.savefig("output.png", dpi=300, wspace=0.05)

# create a dataset with baselines
np.random.seed(42)
N_vis = 50
data_points = np.random.uniform(low=0.9 * np.min(vs), high=0.9 * np.max(vs), size=(N_vis, 2))


# it may help to first sort the u,v datapoints according to v, so that we see more of them.

u_data, v_data = data_points.T

print(data_points)
print(us)
print(vs)


fig, ax = plt.subplots(nrows=1)
ax.scatter(u_data, v_data)
fig.savefig("baselines.png", dpi=300)

# calculate and visualize the C_real and C_imag matrices
import gridding

C_real, C_imag = gridding.calc_matrices(data_points, us, vs)

print(C_real.shape)

fig, ax = plt.subplots(ncols=1, figsize=(12,3))
# ax[0].imshow(C_real[], interpolation="none", origin="upper")
# ax[1].imshow(C_imag[], interpolation="none", origin="upper")
ax.imshow(C_real[:,0:500], interpolation="none", origin="upper")
fig.savefig("C_real.png", dpi=300)

fig, ax = plt.subplots(ncols=1, figsize=(12,3))
ax.imshow(C_imag[:,0:500], interpolation="none", origin="upper")
fig.savefig("C_imag.png", dpi=300)

# FIX the edge-cases

# use these to interpolate the RFFT output

# compare the interpolated output to the real Gaussian values.