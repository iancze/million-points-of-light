# A Million Points of Light
A few orders of magnitude more than a thousand.

This is the code repository for "A Million Points of Light" or ``MPoL`` for short. Then name comes from the total number of pixels in a typical ALMA datacube these days. The (work in progress) paper is available here: [[source](https://github.com/iancze/million-points-of-tex)]

![Logo](logo.png)

Copyright Ian Czekala, Dan Foreman-Mackey, and collaborators 2018-

email: iancze@gmail.com


# Motivation

This is an attempt to bridge the gulf between the amazing quality ALMA data coming out (e.g., DSHARP) with some of the great (magneto-)hydrodynamical simulations of protoplanetary disks with something other than the crude parametric models we observers typically cook up (e.g., see Czekala et al. recent years for some pretty basic examples).

*Briefly, the main functionality of this package that you may be most interested in is the ability to calculate the visibility likelihood and its gradients for an arbitrarily complex sky-plane intensity model.*

**First, why the visibility domain?** Interferometers measure the Fourier transform of the sky brightness, typically called the visibility function. It is sampled (noisily) at discrete spatial frequencies corresponding to the baseline separations between antennas in the interferometric array. Usually we synthesize and deconvolve images (using algorithms like CLEAN) to recover a model of the sky brightness of our target. However, because the array might not capture some missing spatial frequencies and CLEAN is a non-linear process, one should always be cognizant of the fact that the CLEANed image might be a limited (or distorted, in the worst case) representation of reality. Also, depending on the array configuration, the image may have some peculiar (correlated) noise characteristics. If the forward-model can be carried all the way to the Fourier domain, then the likelihood function is a straightforward Gaussian (chi^2) with the uncertainties provided by the thermal visibility weights.

**Second, why the gradient?** Forward-modeling to the visibility plane is nothing new; radio astronomers have been doing this since the very beginning: at first, with analytic models like Gaussians and rings and later using the FFT to bring arbitrary images to the Fourier plane. The interpolation from a regular grid (FFT output) to the discrete (u,v) baselines should be done using convolutional algorithms to prevent sidelobes and aliasing (e.g., using something like Schwab 1984 and not a linear or nearest neighbor interpolation). What's new here is that we've connected the whole process of image specification, FFT, and (u,v) interpolation through an autodifferentiation framework (called Theano). This means that for a given set of model parameters, we can calculate not only the likelihood function, but also its gradient with respect to the parameters. Utilizing the gradient information makes exploring high dimensional probability surfaces much easier, especially with MCMC algorithms like Hamiltonian Monte Carlo. Such a framework should be able to easily converge to the target distribution for models with 20 - 30 parameters.

**Third, what did you mean by "arbitrarily complex?"**
As long as you can specify your image-plane intensity model using [Theano functions](http://deeplearning.net/software/theano/tutorial/index.html#tutorial), which cover most of the routines you're probably interested in, like `sin`, `exp`, etc... then Theano can calculate the gradients of your model. For example, you might construct a model that has a series of rings (inclined as ellipses), with thickness set by some Gaussian taper. Alternatively, you could make a logarithmic spiral---there's no restriction that the model be axisymmetric or any other restrictions for that matter, as long as you can implement it with Theano functions. (And, if you can't, you can always consider implementing the function you need as a Theano `op`).

# An example
