#!/beegfs/home/liuming/software/install/python/bin/python3
import os
import numpy as np
import cupy as cp
from opt_einsum import contract
# from gamma_matrix_cupy_DR import *

# ------------------------------------------------------------------------------------------------
def Peram_truncated(peram):
    peram_tc=peram[:,0:2,0:2,:,:]
    return peram_tc

# ------------------------------------------------------------------------------
def readin_eigvecs_device(eig_dir, t, Nev1,conf_id, Nx):
    f=open("%s/eigvecs_t%03d_%s"%(eig_dir, t, conf_id),'rb')
    eigvecs=np.fromfile(f,dtype='f8')
    eigvecs_size=eigvecs.size
    Nev=int(eigvecs_size/(Nx*Nx*Nx*3*2))
    eigvecs=eigvecs.reshape(Nev,Nx*Nx*Nx,3,2)
    eigvecs=eigvecs[...,0]+eigvecs[...,1]*1j
    eigvecs=eigvecs[0:Nev1,:,:]
    eigvecs = np.transpose(eigvecs,(1,0))
    eigvecs_cupy=cp.asarray(eigvecs)
    return eigvecs_cupy

# ------------------------------------------------------------------------------
def readin_eigvecs_cpu(eig_dir, t, Nev1,conf_id, Nx):
    f=open("%s/eigvecs_t%03d_%s"%(eig_dir, t, conf_id),'rb')
    eigvecs=np.fromfile(f,dtype='f8')
    eigvecs_size=eigvecs.size
    Nev=int(eigvecs_size/(Nx*Nx*Nx*3*2))
    eigvecs=eigvecs.reshape(Nev,Nx*Nx*Nx,3,2)
    eigvecs=eigvecs[...,0]+eigvecs[...,1]*1j
    eigvecs=eigvecs[0:Nev1,:,:]
    eigvecs = np.transpose(eigvecs,(1,0))
    # eigvecs_cupy=cp.asarray(eigvecs)
    return eigvecs

# ------------------------------------------------------------------------------------------------
# Based on "readin_peram_v2", read-in perambulator from one particular timeslice, and saved it in device memory
def readin_peram_device(peram_dir, conf_id, Nt, Nev1, t_source):
    f=open("%s/perams.%s.0.%i"%(peram_dir,conf_id,t_source),'rb')
    peram=np.fromfile(f,dtype='f8')
    f.close()
    
    for d_source in range(1,4):
        f=open("%s/perams.%s.%i.%i"%(peram_dir,conf_id,d_source,t_source),'rb')
        temp=np.fromfile(f,dtype='f8')
        peram=np.append(peram, temp)
        temp=None
        f.close()
    peram_size=peram.size
    Nev=int(np.sqrt(peram_size/(4*4*Nt*2)))
    peram=peram.reshape(4,Nt,Nev,4,Nev,2) #d_source, t_sink, ev_source, d_sink, ev_sink, complex
    peram=peram.transpose(1,3,0,4,2,5) #t_sink, d_sink, d_source, ev_sink, ev_souce,  complex
    peram=peram[...,0] + peram[...,1]*1j
    peram=peram[:,:,:,0:Nev1,0:Nev1]
    peram_cupy=cp.array(peram)
    return peram_cupy

# ------------------------------------------------------------------------------------------------
# Basically identical to v2, except the data is stored on RAM
def readin_peram_all_cpu(peram_dir, conf_id, Nt, Nev1): # One less parameter compared to pervious verison
    peram_cpu_all=np.zeros((Nt,Nt,4,4,Nev1,Nev1),dtype=complex) #t_source, t_sink, d_sink, d_source, ev_sink, ev_source,  complex
    for t_source in range(0,Nt):
        f=open("%s/perams.%s.0.%i"%(peram_dir,conf_id,t_source),'rb')
        peram=np.fromfile(f,dtype='f8')
        f.close()
        
        for d_source in range(1,4):
            f=open("%s/perams.%s.%i.%i"%(peram_dir,conf_id,d_source,t_source),'rb')
            temp=np.fromfile(f,dtype='f8')
            peram=np.append(peram, temp)
            temp=None
            f.close()
        peram_size=peram.size
        Nev=int(np.sqrt(peram_size/(4*4*Nt*2)))
        peram=peram.reshape(4,Nt,Nev,4,Nev,2) #d_source, t_sink, ev_source, d_sink, ev_sink, complex
        peram=peram.transpose(1,3,0,4,2,5) #t_sink, d_sink, d_source, ev_sink, ev_source,  complex
        peram=peram[...,0] + peram[...,1]*1j
        peram_cpu_all[t_source]=peram[:,:,:,0:Nev1,0:Nev1]
    # peram_cupy=cp.array(peram)
    return peram_cpu_all

# ------------------------------------------------------------------------------------------------
def readin_VVV_device(VVV_dir, Nev1, t, conf_id, Px, Py, Pz):
    VVV=np.zeros((Nev1, Nev1, Nev1),dtype=complex)
    f=open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s"%(VVV_dir, t, Px, Py, Pz, conf_id),'rb')
    temp=np.fromfile(f,dtype='f8')
    Nev=int(np.cbrt(temp.size/2))
    temp=temp.reshape(Nev,Nev,Nev,2)
    temp=temp[...,0]+temp[...,1]*1j
    temp=temp[0:Nev1,0:Nev1,0:Nev1]
    VVV=np.copy(temp)
    f.close()
    VVV_cupy=cp.array(VVV)
    
    return VVV_cupy

# ------------------------------------------------------------------------------------------------
def readin_VVV_cpu(VVV_dir, Nev1, t, conf_id, Px, Py, Pz):
    VVV=np.zeros((Nev1, Nev1, Nev1),dtype=complex)  
    f=open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s"%(VVV_dir, t, Px, Py, Pz, conf_id),'rb')
    temp=np.fromfile(f,dtype='f8')
    Nev=int(np.cbrt(temp.size/2))
    temp=temp.reshape(Nev,Nev,Nev,2)
    temp=temp[...,0]+temp[...,1]*1j
    temp=temp[0:Nev1,0:Nev1,0:Nev1]
    VVV=np.copy(temp)
    f.close()
    # VVV_cupy=cp.array(VVV)
    return VVV

# ------------------------------------------------------------------------------------------------
def readin_VVV_all_device(VVV_dir, Nev1, Nt, conf_id, Px, Py, Pz):
  VVV=np.zeros((Nt, Nev1, Nev1, Nev1),dtype=complex)
  for t in range(0,Nt):
    f=open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s"%(VVV_dir, t, Px, Py, Pz, conf_id),'rb')
    temp=np.fromfile(f,dtype='f8')
    Nev=int(np.cbrt(temp.size/2))
    temp=temp.reshape(Nev,Nev,Nev,2)
    temp=temp[...,0]+temp[...,1]*1j
    temp=temp[0:Nev1,0:Nev1,0:Nev1]
    VVV[t]=np.copy(temp)
    f.close()
  VVV_cupy=cp.array(VVV)
  return VVV_cupy

# ------------------------------------------------------------------------------------------------
# Basically identical to original, except VVV is saved on the host, not on the device.
def readin_VVV_all_cpu(VVV_dir, Nev1, Nt, conf_id, Px, Py, Pz):
  VVV=np.zeros((Nt, Nev1, Nev1, Nev1),dtype=complex)
  for t in range(0,Nt):
    f=open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s"%(VVV_dir, t, Px, Py, Pz, conf_id),'rb')
    temp=np.fromfile(f,dtype='f8')
    Nev=int(np.cbrt(temp.size/2))
    temp=temp.reshape(Nev,Nev,Nev,2)
    temp=temp[...,0]+temp[...,1]*1j
    temp=temp[0:Nev1,0:Nev1,0:Nev1]
    VVV[t]=np.copy(temp)
    f.close()
  # VVV_cupy=cp.array(VVV)
  return VVV

# ------------------------------------------------------------------------------------------------
def readin_VdV_all_device(VdV_dir, Nev1, Nt, conf_id, Px, Py, Pz):
  f=open("%s/VdaggerV.Px%dPy%dPz%d.conf%s"%(VdV_dir, Px, Py, Pz, conf_id),'rb')
  VdV=np.fromfile(f,dtype='f8')
  Nev=int(np.sqrt(VdV.size/(Nt*2)))
  VdV=VdV.reshape(Nt, Nev, Nev, 2)
  VdV=VdV[...,0] + VdV[...,1]*1j
  VdV=VdV[:,0:Nev1, 0:Nev1]
  VdV_cupy=cp.array(VdV)
  f.close()
  return VdV_cupy	

# ------------------------------------------------------------------------------------------------
def readin_VdV_all_cpu(VdV_dir, Nev1, Nt, conf_id, Px, Py, Pz):
  f=open("%s/VdaggerV.Px%dPy%dPz%d.conf%s"%(VdV_dir, Px, Py, Pz, conf_id),'rb')
  VdV=np.fromfile(f,dtype='f8')
  Nev=int(np.sqrt(VdV.size/(Nt*2)))
  VdV=VdV.reshape(Nt, Nev, Nev, 2)
  VdV=VdV[...,0] + VdV[...,1]*1j
  VdV=VdV[:,0:Nev1, 0:Nev1]
  f.close()
#   VdV_cupy=cp.array(VdV)
  return VdV

# ------------------------------------------------------------------------------------------------
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
       savetxt(filename, _fdata, header=head, comments='', fmt=["%i", "%.32f", "%.32f"])
    else:
       head = "%i %i %i %i %i" % (nsamples, T, 0, L, 1)
       _fdata = np.concatenate((_counter, _data), axis=1)
       savetxt(filename, _fdata, header=head, comments='', fmt=["%i", "%.32f"])

# ------------------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------------
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
