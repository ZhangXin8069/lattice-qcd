#!/public/home/xinghy/anaconda3-2023.03/envs/cupy114/bin/python
import os
import numpy as np
def write_data_ascii(data, T, L, filename, complex=True, verbose=False):
    """Writes the data into a file.
    The file is written to have L. Liu's data format so that the first line
    has information about the number of samples and the length of each sample.
    Args:
        filename: The filename of the file.
        data: The numpy array with data.
        verbose: The amount of info shown.
    """
    # check file
    check_write(filename)
    if verbose:
        print("saving to file " + str(filename))
    # in case the dimension is 1, treat the data as one sample
    # to make the rest easier we add an extra axis
    nsamples = data.shape[0]
#    T = data.shape[1]
#    L = int(T/2) 
    _data = data.reshape((T*nsamples), -1)
    _counter = np.fromfunction(lambda i, *j: i%T,
                               (_data.shape[0],) + (1,)*(len(_data.shape)-1),
                               dtype=int)
    if complex:
       head = "%i %i %i %i %i" % (nsamples, T, 1, L, 1)
       data_real = _data.real
       data_imag = _data.imag
       _fdata = np.concatenate((_counter, data_real, data_imag), axis=1)
       savetxt(filename, _fdata, header=head, comments='', fmt=["%i", "%.16e", "%.16e"])
    else:
       head = "%i %i %i %i %i" % (nsamples, T, 0, L, 1)
       _fdata = np.concatenate((_counter, _data), axis=1)
       savetxt(filename, _fdata, header=head, comments='', fmt=["%i", "%.16e"])
def check_write(filename):
    """Do some checks before writing a file.
    """
    # check if path exists, if not then create it
    _dir = os.path.dirname(filename)
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    # check whether file exists
    if os.path.isfile(filename):
        print(filename + " already exists, overwritting...")
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
                footer='', comments='# '):
    """This code is from NumPy 1.9.1. For help see there.
    It was included because features are used that were added in version 1.7
    but on some machines only NumPy version 1.6.2 is available.
    """
    ## needed for the rest
    from numpy.compat import asstr, asbytes
    def _is_string_like(obj):
        try:
            obj + ''
        except (TypeError, ValueError):
            return False
        return True
    # Py3 conversions first
    if isinstance(fmt, bytes):
        fmt = asstr(fmt)
        delimiter = asstr(delimiter)
    own_fh = False
    if _is_string_like(fname):
        own_fh = True
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname, 'wb')
        else:
            if os.sys.version_info[0] >= 3:
                fh = open(fname, 'wb')
            else:
                fh = open(fname, 'w')
    elif hasattr(fname, 'write'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
    try:
        X = np.asarray(X)
        # Handle 1-dimensional arrays
        if X.ndim == 1:
            # Common case -- 1d array of numbers
            if X.dtype.names is None:
                X = np.atleast_2d(X).T
                ncol = 1
            # Complex dtype -- each field indicates a separate column
            else:
                ncol = len(X.dtype.descr)
        else:
            ncol = X.shape[1]
        iscomplex_X = np.iscomplexobj(X)
        # `fmt` can be a string with multiple insertion points or a
        # list of formats.  E.g. '%10.5f\t%10d' or ('%10.5f', '%10d')
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
            format = asstr(delimiter).join(map(asstr, fmt))
        elif isinstance(fmt, str):
            n_fmt_chars = fmt.count('%')
            error = ValueError('fmt has wrong number of %% formats:  %s' % fmt)
            if n_fmt_chars == 1:
                if iscomplex_X:
                    fmt = [' (%s+%sj)' % (fmt, fmt), ] * ncol
                else:
                    fmt = [fmt, ] * ncol
                format = delimiter.join(fmt)
            elif iscomplex_X and n_fmt_chars != (2 * ncol):
                raise error
            elif ((not iscomplex_X) and n_fmt_chars != ncol):
                raise error
            else:
                format = fmt
        else:
            raise ValueError('invalid fmt: %r' % (fmt,))
        if len(header) > 0:
            header = header.replace('\n', '\n' + comments)
            fh.write(asbytes(comments + header + newline))
        if iscomplex_X:
            for row in X:
                row2 = []
                for number in row:
                     row2.append(number.real)
                     row2.append(number.imag)
                fh.write(asbytes(format % tuple(row2) + newline))
        else:
            for row in X:
                fh.write(asbytes(format % tuple(row) + newline))
        if len(footer) > 0:
            footer = footer.replace('\n', '\n' + comments)
            fh.write(asbytes(comments + footer + newline))
    finally:
        if own_fh:
            fh.close()
