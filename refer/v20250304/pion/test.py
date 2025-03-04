# from mpi4py import MPI
# import numpy as np
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# if rank == 0:
#     send_data = np.arange(16).reshape(4,4)
# else:
#     send_data = np.arange(16).reshape(4,4)
#     sendshape = 0
#     B = None
# # recv_data = comm.bcast(C,root=0)
# recv_data = comm.gather(send_data, root=0)
# # # 打印结果
# if rank == 0:
#     print("Root process (rank {}):".format(rank))
#     print("Received data:", np.asarray(recv_data).reshape(8,4))
# if rank == 1:
#     print("Root process (rank {}):".format(rank))
#     print("Received data:", recv_data)
# from mpi4py import MPI 
# import numpy as np 
# import psutil
# import os
# comm = MPI.COMM_WORLD 
# rank = comm.Get_rank()
# size_rank = comm.Get_size()
# Lt = 72 // size_rank
# n = 2
# # create a shared array of size 1000 elements of type double
# size = 72*24*24*24*3*3
# itemsize = MPI.DOUBLE.Get_size() 
# if comm.Get_rank() == 0: 
#     nbytes = size * itemsize * n
# else: 
#     nbytes = 0
# # on rank 0, create the shared block
# # on rank 1 get a handle to it (known as a window in MPI speak)
# win = MPI.Win.Allocate_shared(nbytes, itemsize * n, comm=comm) 
# # create a numpy array whose data points to the shared mem
# buf, itemsize = win.Shared_query(0) 
# assert itemsize == MPI.DOUBLE.Get_size() * n
# ary = np.ndarray(buffer=buf, dtype=np.complex128, shape=(72,24,24,24,3,3))
# # in process rank 1:
# # write the numbers 0.0,1.0,..,4.0 to the first 5 elements of the array
# for i in range(Lt):
#     ary[i + Lt*rank] = np.arange(size // 72, dtype=np.complex128).reshape(24,24,24,3,3)
# # wait in process rank 0 until process 1 has written to the array
# comm.Barrier() 
# # check that the array is actually shared and process 0 can see
# # the changes made in the array by process 1
# if comm.rank == 1: 
#     print(ary[0])
# comm.Barrier() 
# print(ary.nbytes)
# print(ary.shape[0])
# print(u'当前程序占用内存：%.4f GB'%(psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024))
import numpy as np
A = np.range()