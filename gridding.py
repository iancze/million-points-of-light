import numpy as np

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

@np.vectorize
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

        return horner(nn, np.array([8.203343E-2, -3.644705E-1, 6.278660E-1, -5.335581E-1, 2.312756E-1])) / horner(nn, np.array([1., 8.212018E-1, 2.078043E-1]))

    elif (eta <= 1.0):
        nn = eta**2 - 1.0

        return horner(nn, np.array([4.028559E-3, -3.697768E-2, 1.021332E-1, -1.201436E-1, 6.412774E-2])) / horner(nn, np.array([1., 9.599102E-1, 2.918724E-1]))

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


def corrfun_mat(alphas, deltas):
    '''
    Calculate the pre-multiply correction function to the image.
    Return as a 2D array.

    Args:
        alphas (1D array): RA list (pre-fftshifted)
        deltas (2D array): DEC list (pre-fftshifted)

    '''

    ny = len(deltas)
    nx = len(alphas)

    mat = np.empty((ny, nx), dtype=np.float64)

    # The size of one half-of the image.
    # sometimes ra and dec will be symmetric about 0, othertimes they won't
    # so this is a more robust way to determine image half-size
    maxra = np.abs(alphas[2] - alphas[1]) * nx/2
    maxdec = np.abs(deltas[2] - deltas[1]) * ny/2

    for i in range(nx):
        for j in range(ny):
            etax = (alphas[i])/maxra
            etay = (deltas[j])/maxdec

            if (np.abs(etax) > 1.0) or (np.abs(etay) > 1.0):
                # We would be querying outside the shifted image
                # bounds, so set this emission to 0.0
                mat[j, i] = 0.0
            else:
                mat[j, i] = 1 / (corrfun(etax) * corrfun(etay))

    return mat

def gcffun(eta):
    '''
    The gridding *convolution* function, used to do the convolution and interpolation of the visibilities in
    the Fourier domain. This is also the Fourier transform of `corrfun`.

    Args:
        eta (float): in the domain of [0,1]
    '''

    return np.abs(1.0 - eta**2) * spheroid(eta)


def calc_matrices(data_points, u_model, v_model):
    '''
    Calcuate the real and imaginary interpolation matrices in one pass.

    Args:
        data_points: the pairs of u,v points in the dataset (in klambda)
        u_model: the u axis delivered by the rfft (unflattened). Assuming this is the RFFT axis.
        v_model: the v axis delivered by the rfft (unflattened). Assuming this is the FFT axis.

    Start with an image packed like Img[j, i]. i is the alpha index and j is the delta index.
    Then the RFFT output will have RFFT[j, i]. i is the u index and j is the v index.

    see also `Model.for` routine in MIRIAD source code.
    Uses spheroidal wave functions to interpolate a model to a (u,v) coordinate.
    Ensure that datapoints and u_model,v_model are in consistent units (either λ or kλ).

    (m, ..., n//2+1, 2)
    u freqs stored like this
    f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even

    v freqs stored like this
    f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even

    '''

    #TODO: assert that the maximum baseline is contained within the model grid.
    #TODO: assert that the image-plane pixels are small enough.

    # number of visibility points in the dataset
    N_vis = len(data_points)

    # calculate the stride needed to advance one v point in the flattened array = the length of the u row
    vstride = len(u_model)
    Npix = len(v_model)

    # initialize two sparse matrices. For now, just call them zeros.
    C_real = np.zeros((N_vis, (Npix * vstride)), dtype=np.float64)
    C_imag = np.zeros((N_vis, (Npix * vstride)), dtype=np.float64)

    # determine model grid spacing
    du = np.abs(u_model[1] - u_model[0])
    dv = np.abs(v_model[1] - v_model[0])

    # for each data_point within the grid, calculate the row and insert it into the matrix
    for row_index, (u, v) in enumerate(data_points):

        if (u > 0) and (np.abs(u) > 3 * du):
            # calculate for +u, with no overlap, which is the easiest

            # if v overlaps, need to split the indices between negative and positive frequencies
            if np.abs(v) < (3 * dv): # v overlaps 0 border
                if (v > 0):
                    j0 = np.searchsorted(v_model[:Npix//2], v) # only search the positive frequencies
                    j_indices = np.arange(j0 - 3, j0 + 3) # 6 points [j0-3,j0-2,j0-1,j0,j0+1,j0+2]
                    j_indices[j_indices < 0] += Npix # those less than 0 get Npix added to them

                else: #(v < 0)
                    j0 = np.searchsorted(v_model[Npix//2:], v) + Npix//2 # only search the negative frequencies
                    j_indices = np.arange(j0 - 3, j0 + 3) # 6 points [j0-3,j0-2,j0-1,j0,j0+1,j0+2]
                    j_indices[j_indices >= Npix] -= Npix # those greater than Npix get Npix subtracted from to them

            else: # no v overlap w/ 0 border
                if (v > 0):
                    j0 = np.searchsorted(v_model[:Npix//2], v) # only search the positive frequencies
                else: #(v < 0)
                    j0 = np.searchsorted(v_model[Npix//2:], v) + Npix//2 # only search the negative frequencies

                j_indices = np.arange(j0 - 3, j0 + 3) # 6 points [j0-3,j0-2,j0-1,j0,j0+1,j0+2]

            v_etas = (v - v_model[j_indices]) / (3 * dv)
            vw = gcffun(v_etas)
            vw_real = vw
            vw_imag = vw

            # calculate the u values
            # find the nearest points in the array
            i0 = np.searchsorted(u_model, u)
            i_indices = np.arange(i0 - 3, i0 + 3) # 6 points [i0-3,i0-2,i0-1,i0,i0+1,i0+2]

            # assemble a list of l indices into the flattened RFFT output
            l_indices = np.array([i + j * vstride for i in i_indices for j in j_indices]) # list of 36 l indices

            # calculate the u and v distances from the (u, v) datapoint to each of the l indices as a function of
            # eta in the domain 0 - 1
            # the 3 is because we have 3 points on either side
            u_etas = (np.abs(u) - u_model[i_indices]) / (3 * du)

            # evaluate the spheroid here
            uw = gcffun(u_etas)

            # Normalization such that it has an area of 1. Divide by w later.
            w = sum(uw) * sum(vw)

            # actual weight at a point is uw[i] * vw[j] / w
            # arrange the uw and vw weights in the same order as ls
            weights_real = np.array([uw[i] * vw_real[j] for i in range(6) for j in range(6)]) / w
            weights_imag = np.array([uw[i] * vw_imag[j] for i in range(6) for j in range(6)]) / w

            # insert this into the C matrix at the corresponding locations (the ls)
            C_real[row_index,l_indices] = weights_real
            C_imag[row_index,l_indices] = weights_imag


        elif (u < 0) and (np.abs(u) > 3 * du):
            # Calculate for -u, no overlap. This means we need to calculate at -v instead
            # (u < 0, v > 0 ) == *(u > 0, v < 0)
            # (u < 0, v < 0 ) == *(u > 0, v > 0)
            v_temp = -v

            # if v overlaps, need to split the indices between negative and positive frequencies
            if np.abs(v_temp) < (3 * dv): # v overlaps 0 border
                if (v_temp > 0):
                    j0 = np.searchsorted(v_model[:Npix//2], v_temp) # only search the positive frequencies
                    j_indices = np.arange(j0 - 3, j0 + 3) # 6 points [j0-3,j0-2,j0-1,j0,j0+1,j0+2]
                    j_indices[j_indices < 0] += Npix # those less than 0 get Npix added to them

                else: #(v < 0)
                    j0 = np.searchsorted(v_model[Npix//2:], v_temp) + Npix//2 # only search the negative frequencies
                    j_indices = np.arange(j0 - 3, j0 + 3) # 6 points [j0-3,j0-2,j0-1,j0,j0+1,j0+2]
                    j_indices[j_indices >= Npix] -= Npix # those greater than Npix get Npix subtracted from to them

            else: # no v overlap w/ 0 border
                if (v_temp > 0):
                    j0 = np.searchsorted(v_model[:Npix//2], v_temp) # only search the positive frequencies
                else: #(v < 0)
                    j0 = np.searchsorted(v_model[Npix//2:], v_temp) + Npix//2 # only search the negative frequencies

                j_indices = np.arange(j0 - 3, j0 + 3) # 6 points [j0-3,j0-2,j0-1,j0,j0+1,j0+2]

            v_etas = (v_temp - v_model[j_indices]) / (3 * dv)
            vw = gcffun(v_etas)
            vw_real = vw
            vw_imag = -vw

            # calculate the u values
            # find the nearest points in the array
            i0 = np.searchsorted(u_model, np.abs(u))
            i_indices = np.arange(i0 - 3, i0 + 3) # 6 points [i0-3,i0-2,i0-1,i0,i0+1,i0+2]

            # assemble a list of l indices into the flattened RFFT output
            l_indices = np.array([i + j * vstride for i in i_indices for j in j_indices]) # list of 36 l indices

            # calculate the u and v distances from the (u, v) datapoint to each of the l indices as a function of
            # eta in the domain 0 - 1
            # the 3 is because we have 3 points on either side
            u_etas = (np.abs(u) - u_model[i_indices]) / (3 * du)

            # evaluate the spheroid here
            uw = gcffun(u_etas)

            # Normalization such that it has an area of 1. Divide by w later.
            w = sum(uw) * sum(vw)

            # actual weight at a point is uw[i] * vw[j] / w
            # arrange the uw and vw weights in the same order as ls
            weights_real = np.array([uw[i] * vw_real[j] for i in range(6) for j in range(6)]) / w
            weights_imag = np.array([uw[i] * vw_imag[j] for i in range(6) for j in range(6)]) / w

            # insert this into the C matrix at the corresponding locations (the ls)
            C_real[row_index,l_indices] = weights_real
            C_imag[row_index,l_indices] = weights_imag

        elif (np.abs(u) < 3 * du):
            # u is sufficiently close to zero that we need to query both - and + u values from the grid
            # so we need to identify two sets of j indices, for - and + uvalues.
            # as we move through the grid, use the value that corresponds to the correct u sign.

            


            # u overlap
            # we'll start out with 36 grid interpolation points, but because of the reflective mirroring,
            # we'll actually be querying the same point twice with different weights,
            # so, these weights should add.

            # print()
            # print("u overlap", row_index, u, "du", du, "v", v)
            #
            # # 1) calculate the 6 distances (eta_us) between the current u point and the adjacent values
            # i0 = np.searchsorted(u_model, np.abs(u))
            #
            # # Can delete later
            # distance_right = i0 - np.abs(u)/du # towards increasing u values
            # print("distance_right", distance_right)
            #
            # # 4) Figure out how many unique i-values we will have after overlap
            # # i_indices was originally calculated assuming that all u values were either negative or positive.
            # # but since they straddle, we need to separate them here to rearrange the order of the u_etas
            # if u > 0:
            #     i_indices = np.arange(i0 - 3, i0 + 3) # 6 points
            #     u_etas = (i_indices - u/du) / 3
            # else:
            #     i_indices = np.arange(-i0 - 2, -i0 + 4) # 6 points
            #     u_etas = (i_indices - u/du) / 3
            #
            # print("u_etas", u_etas)
            #
            # # 2) calculate the 6 u-weights
            # uw = gcffun(u_etas)
            # print("uw", uw)
            #
            # # 3) use these, with the v-values, to calculate the normalization w
            # w = sum(uw) * sum(vw)
            #
            #
            # print("i_indices", i_indices)
            # n_unique = np.max(np.abs(i_indices)) + 1 # 1 is to count 0 index as well
            # print("n_unique", n_unique)
            #
            # ind_pos = (i_indices >= 0)
            # n_pos = np.sum(ind_pos)
            # print("ind_pos", ind_pos)
            # print("n_pos", n_pos)
            #
            # # 5) Create a shortened storage array
            # uw_collapsed_real = np.zeros(n_unique)
            # uw_collapsed_imag = np.zeros(n_unique)
            #
            # # insert the positive values
            # uw_collapsed_real[:n_pos] = uw[ind_pos]
            # uw_collapsed_imag[:n_pos] = uw[ind_pos]
            #
            # print("uw_collapsed_real", uw_collapsed_real)
            # print("uw_collapsed_imag", uw_collapsed_imag)
            #
            # # 6) find the indices of the negative values
            # ind_neg = ~ind_pos
            # position_neg = np.abs(i_indices[ind_neg])
            # print(position_neg, "position_neg")
            #
            # # add them to the existing weights
            # uw_collapsed_real[position_neg] += uw[ind_neg]
            # uw_collapsed_imag[position_neg] += -uw[ind_neg] # complex conjugate for imaginary values
            # print("uw_collapsed_real", uw_collapsed_real)
            # print("uw_collapsed_imag", uw_collapsed_imag)
            #
            #
            # # For every negative u point, we actually need to be querying the complex conjugate of the
            # # -v point as well.
            #
            # # 7) assemble a list of l indices into the flattened RFFT output
            # # note that we're looping over range(n_unique) rather than i_indices
            # l_indices = np.array([i + j * vstride for i in range(n_unique) for j in j_indices]) # list of 36 l indices
            #
            # weights_real = np.array([uw_collapsed_real[i] * vw[j] for i in range(n_unique) for j in range(6)]) / w
            # weights_imag = np.array([uw_collapsed_imag[i] * vw[j] for i in range(n_unique) for j in range(6)]) / w

            # C_real[row_index, l_indices] = weights_real
            # C_imag[row_index, l_indices] = weights_imag

            C_real[row_index, l_indices] = np.nan
            C_imag[row_index, l_indices] = np.nan

        else:
            print("Shouldn't have gotten here.")
            raise

    return C_real, C_imag


# since we will probably have the case that len(data_points) > (len(u_model) * len(v_model)),
# it is more efficient to use a compressed sparse column (row) matrix
# according to Theano,
# If shape[0] > shape[1], use csc format. Otherwise, use csr.
