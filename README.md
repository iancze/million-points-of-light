# A Million Points of Light
A few orders of magnitude more than a thousand.

This is the code repository for "A Million Points of Light" or ``MPoL`` for short. Then name comes from the total number of pixels in a typical ALMA datacube these days. The (work in progress) paper is available here: [[PDF](https://v2.overleaf.com/project/5bdafa78941fc514888e7548/output/output.pdf?compileGroup=priority&clsiserverid=clsi-pre-emp-kx7q&popupDownload=true)] [[source](https://github.com/iancze/million-points-of-tex)]

![Logo](logo.png)

Copyright Ian Czekala and collaborators 2018-

email: iancze@gmail.com


# Motivation

This is an attempt to bridge the gulf between the amazing quality ALMA data coming out (e.g., DSHARP) with some of the great (magneto-)hydrodynamical simulations of protoplanetary disks with something other than the crude parametric models we observers typically cook up (e.g., see Czekala et al. recent years for some pretty basic examples).

*Briefly, the main functionality of this package that you may be most interested in is the ability to calculate the visibility likelihood and its gradients for an arbitrarily complex sky-plane intensity model.*

**First, why the visibility domain?** Interferometers measure the Fourier transform of the sky brightness, typically called the visibility function. It is sampled (noisily) at discrete spatial frequencies corresponding to the baseline separations between antennas in the interferometric array. Usually we synthesize and deconvolve images (using algorithms like CLEAN) to recover a model of the sky brightness of our target. However, because the array might not capture some missing spatial frequencies and CLEAN is a non-linear process, one should always be cognizant of the fact that the CLEANed image might be a limited (or distorted, in the worst case) representation of reality. Also, depending on the array configuration, the image may have some peculiar (correlated) noise characteristics.


**Second, why the gradient?**

**Third, what do you mean by "arbitrarily complex?"**


# An example
