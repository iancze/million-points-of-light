# implementation of the gridding convolution functions and image pre-multiply

# fill the sparse interpolation matrix.

def horner(x, a):
    '''
    Use Horner's method to compute and return the polynomial
        a[0] + a[1] x^1 + a[2] x^2 + ... + a[n-1] x^(n-1)
    evaluated at x.

    from https://introcs.cs.princeton.edu/python/21function/horner.py.html
    '''
    result = 0
    for i in range(len(a)-1, -1, -1):
        result = a[i] + (x * result)
    return result

def spheroid(eta):
    '''
        `spheroid` function which assumes ``\\alpha`` = 1.0, ``m=6``,  built for speed."

        Args:
            eta (float) : the value between [0, 1]
    '''

    # Since the function is symmetric, overwrite eta
    eta = np.abs(eta)

    if (eta <= 0.75):
        nn = eta**2 - 0.75**2

        return horner(nn, np.array([8.203343E-2, -3.644705E-1, 6.278660E-1, -5.335581E-1, 2.312756E-1]))/
            horner(nn, np.array([1., 8.212018E-1, 2.078043E-1]))

    elif (eta <= 1.0):
        nn = eta**2 - 1.0

        return horner(nn, np.array([4.028559E-3, -3.697768E-2, 1.021332E-1, -1.201436E-1, 6.412774E-2]))/
            horner(nn, np.array([1., 9.599102E-1, 2.918724E-1]))

    elif (eta <= 1.0 + 1e-7):
        # case to allow some floating point error
        return 0.0

    else:
        # Now you're really outside of the bounds
        print("The spheroid is only defined on the domain -1.0 <= eta <= 1.0. (modulo machine precision.)")
        raise ValueError


def corrfun(eta):
    '''
    Gridding *correction* function, but able to be passed either floating point numbers or vectors of `Float64`."

    Args:
        eta (float): the value in [0, 1]
    '''
    return spheroid(eta)


# def corrfun(img):
#     '''
#     Calculate the pre-multiply correction function to the image.
#     '''
#     ny, nx, nlam = size(img.data)
#
#     # The size of one half-of the image.
#     # sometimes ra and dec will be symmetric about 0, othertimes they won't
#     # so this is a more robust way to determine image half-size
#     maxra = abs(img.ra[2] - img.ra[1]) * nx/2
#     maxdec = abs(img.dec[2] - img.dec[1]) * ny/2
#
#     for k=1:nlam
#         for i=1:nx
#             for j=1:ny
#                 etax = (img.ra[i])/maxra
#                 etay = (img.dec[j])/maxdec
#                 if abs(etax) > 1.0 || abs(etay) > 1.0
#                     # We would be querying outside the shifted image
#                     # bounds, so set this emission to 0.0
#                     img.data[j, i, k] = 0.0
#                 else
#                     img.data[j, i, k] = img.data[j, i, k] / (corrfun(etax) * corrfun(etay))
#                 end
#             end
#         end
#     end
# end


def gcffun(eta):
    '''
    The gridding *convolution* function, used to do the convolution and interpolation of the visibilities in
    the Fourier domain. This is also the Fourier transform of `corrfun`.

    Args:
        eta (float): in the domain of [0,1]
    '''

    return np.abs(1.0 - eta**2) * spheroid(eta)


# assume that the rfft receives an image packed like
# [a0, d0], [a1,d0],
# [a0, d1], [a1,d1]
# i.e., like traditional images, the row index corresponds to y
# and the column index corresponds to x.

# then the alphas (or u's) are the ones that have been rfft, and the vs are the ones that have been fft'ed
# so the output array will be
# [u0, v0] [u1, v0]
# [u0, v1] [u1, v1]
# [u0, v2] [u1, v2]
# [u0, v3] [u1, v3]

 # (m, ..., n//2+1, 2)
 # u freqs stored like this
 # f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even

 # v freqs stored like this
 # f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even

# so if we flatten this, we'll have
# # [u0, v0] [u1, v0] [u0, v1] [u1, v1] [u0, v2] [u1, v2] [u0, v3] [u1, v3]
# i.e., u0 is going through the full range (pos only) and then v is coming together.
# so we need to get the 6 (3) nearest points in u, taking into account the +/-
# and then find the nearest v points
# since we can't do an FFTshift, we need an alternate stride to get the -v points



# know that the v stride is the number of u points.

def calc_row(u_data, v_data, u_model, v_model):
    '''
    Calcuate the 36 coefficients for the row.

    Args:
        u_data: the u point (in klambda)
        v_data: the v point (in klambda)
        u_model: the array of u points delivered by the rfft
        v_model: the array of v points delivered by the rfft
    '''


# called ModGrid in gridding.c (KR code) and in Model.for (MIRIAD)
# Uses spheroidal wave functions to interpolate a model to a (u,v) coordinate.
# u,v are in [kλ]

"
    interpolate_uv(u::Float64, v::Float64, vis::FullModelVis)
Interpolates a dense grid of visibilities (e.g., from FFT of an image) to a specfic (u,v) point using spheroidal functions in a band-limited manner designed to reduce aliasing.
"
function interpolate_uv(u::Float64, v::Float64, vis::FullModelVis)

    # Note that vis.uu goes from positive to negative (East-West)
    # and vis.vv goes from negative to positive (North-South)

    # 1. Find the nearest gridpoint in the FFT'd image.
    iu0 = argmin(abs.(u .- vis.uu))
    iv0 = argmin(abs.(v .- vis.vv))

    # now find the relative distance from (u,v) to this nearest grid point (not absolute)
    u0 = u - vis.uu[iu0]
    v0 = v - vis.vv[iv0]

    # determine the uu and vv distance for 3 grid points (could be later taken out)
    du = abs.(vis.uu[4] - vis.uu[1])
    dv = abs.(vis.vv[4] - vis.vv[1])

    # 2. Calculate the appropriate u and v indexes for the 6 nearest pixels
    # (3 on either side)

    # Are u0 and v0 to the left or the right of the index?
    # we want to index three to the left, three to the right

    # First check that our (u,v) point still exists within the appropriate margins of the
    # dense visibility array
    # This is to make sure that at least three grid points exist in all directions
    # If this fails, this means that the synthesized image is too large compared to the sampled visibilities (meaning that the dense FFT grid is too small).
    # The max u,v sampled is 2/dRA or 2/dDec. This means that if dRA or dDEC is too large, then this
    # will fail
    lenu = length(vis.uu)
    lenv = length(vis.vv)
    @assert iu0 >= 4
    @assert iv0 >= 4
    @assert lenu - iu0 >= 4
    @assert lenv - iv0 >= 4

    if u0 >= 0.0
        # To the right of the index
        uind = iu0-2:iu0+3
    else
        # To the left of the index
        uind = iu0-3:iu0+2
    end

    if v0 >= 0.0
        # To the right of the index
        vind = iv0-2:iv0+3
    else
        # To the left of the index
        vind = iv0-3:iv0+2
    end

    etau = (vis.uu[uind] .- u)/du
    etav = (vis.vv[vind] .- v)/dv
    VV = vis.VV[vind, uind] # Array is packed like the image

    # 3. Calculate the weights corresponding to these 6 nearest pixels (gcffun)
    uw = gcffun(etau)
    vw = gcffun(etav)

    # 4. Normalization such that it has an area of 1. Divide by w later.
    w = sum(uw) * sum(vw)

    # 5. Loop over all 36 grid indices and sum to find the interpolation.
    cumulative::ComplexF64 = 0.0 + 0.0im
    for i=1:6
        for j=1:6
            cumulative += uw[i] * vw[j] * VV[j,i] # Array is packed like the image
        end
    end

    cumulative = cumulative/w

    return cumulative
end
