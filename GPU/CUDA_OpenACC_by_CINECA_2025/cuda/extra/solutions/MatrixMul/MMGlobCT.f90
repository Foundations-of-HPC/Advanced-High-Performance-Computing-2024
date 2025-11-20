! start the module containing the matmul kernel
module mmul_mod

use cudafor

contains
attributes(global) subroutine mmul_global_kernel( A, B, C, N )
integer, value :: N
real :: A(N,N), B(N,N), C(N,N)
integer :: i, j, k
real :: cvalue 
! This thread computes C(i,j) = sum(A(i,:) * B(:,j))
! INSERT THREADS INDECES
i = (blockidx%x-1) * blockdim%x + threadidx%x
j = (blockidx%y-1) * blockdim%y + threadidx%y
!
if((i.le.N).and.(j.le.N)) then
! INSERT COMPUTATIONAL CORE
  cvalue = 0.
  do k=1,N
     cvalue = cvalue + A(i,k)*B(k,j)
  enddo
  C(i,j) = cvalue
endif
end subroutine mmul_global_kernel

attributes(global) subroutine mmul_global_kernel_slow( A, B, C, N )
integer, value :: N
real :: A(N,N), B(N,N), C(N,N)
integer :: i, j, k
real :: cvalue 
! This thread computes C(i,j) = sum(A(i,:) * B(:,j))
i = (blockidx%x-1) * blockdim%x + threadidx%x
j = (blockidx%y-1) * blockdim%y + threadidx%y
!
if((i.le.N).and.(j.le.N)) then
  C(i,j) = 0.
  do k=1,N
     C(i,j) = C(i,j) + A(i,k)*B(k,j)
  enddo
endif
end subroutine mmul_global_kernel_slow

! The host routine to drive the matrix multiplication
subroutine mmul_device( A, B, C, N )
use cudafor
integer N,ierr
real, dimension(N,N) :: A, B, C
real, device, allocatable, dimension(:,:) :: Adev,Bdev,Cdev
type(dim3) :: dimGrid, dimBlock
type(cudaEvent) :: start_event,end_event
real :: elapsed_cutime
integer :: mycudaerror 
integer grid_dimx,grid_dimy

allocate( Adev(N,N), Bdev(N,N), Cdev(N,N) )
Adev = A
Bdev = B

dimBlock = dim3(32, 32, 1)
! CUDA grid definition
grid_dimx = int(N/dimBlock%x)
grid_dimy = int(N/dimBlock%y)
if(mod(N,dimBlock%x) .gt. 0) grid_dimx = grid_dimx + 1
if(mod(N,dimBlock%y) .gt. 0) grid_dimy = grid_dimy + 1

dimGrid = dim3(grid_dimx,grid_dimy,1)
write(*,*) 'Gridsize :  ',grid_dimx,' x ',grid_dimy

! create start and stop events
! INSERT CODE
ierr = cudaEventCreate(start_event)
ierr = cudaEventCreate(end_event)
ierr = cudaEventRecord(start_event,0)

! cudaGetLastError call to reset previous CUDA error
! INSERT CODE
mycudaerror = cudaGetLastError()  ;

! kernel launch
call mmul_global_kernel<<<dimGrid,dimBlock>>>( Adev, Bdev, Cdev, N)

! device synchronization and cudaGetLastError call
! INSERT CODE
ierr = cudaDeviceSynchronize()
mycudaerror = cudaGetLastError()  ;
if(mycudaerror .ne. CUDASUCCESS)  then
  write(*,*)"Cuda Error in Kernel MatrixMulKernel: ",cudaGetErrorString(mycudaerror)
endif

! event record, synchronization, elapsed time and destruction
! INSERT CODE
ierr = cudaEventRecord(end_event,0)
ierr = cudaEventSynchronize(end_event)
ierr = cudaEventElapsedTime(elapsed_cutime,start_event,end_event)
ierr = cudaEventDestroy(start_event)
ierr = cudaEventDestroy(end_event)

mflops = 2*float(N)*float(N)*float(N)/(1000.0*1000.0)
print*,'CUDA Event elapsed time:  ',elapsed_cutime
print*,'CUDA Mflops:  ',mflops/(elapsed_cutime/1000.)

C = Cdev
deallocate( Adev, Bdev, Cdev )
end subroutine mmul_device
end module mmul_mod

subroutine mmul_host( A, B, C, N )
integer N
real, dimension(N,N) :: A, B, C
integer i,j,k
C = 0.
do j=1,N
do k=1,N
do i=1,N
   C(i,j) = C(i,j) + A(i,k)*B(k,j)
enddo
enddo
enddo
end subroutine mmul_host

program main_mmul

use mmul_mod

real, dimension(:,:),allocatable:: A,B,Ch,Cd
integer, parameter :: N = 1024
integer :: i,j
real, device, allocatable, dimension(:,:) :: Adev,Bdev,Cdev
type(dim3) :: dimGrid, dimBlock
integer cnt_err

mflops = 2*float(N)*float(N)*float(N)/(1000.0*1000.0)

allocate(A(N,N),B(N,N),Ch(N,N),Cd(N,N))

do i=1,N
do j=1,N
   A(i,j) = (i*N+j)/N 
   B(i,j) = (i*N+j)/N 
enddo
enddo

call mmul_device( A, B, Cd, N)

call mmul_host( A, B, Ch, N)

cnt_err = 0
errore = 0.
do j=1,N
do i=1,N
   errore = max(abs(Cd(i,j)-Ch(i,j)),errore)
   if(errore.gt.1e-5) then 
      cnt_err = cnt_err + 1
      write(*,*) i,j,Cd(i,j),Ch(i,j)
      read(*,*)
   endif
enddo
enddo
if (cnt_err .eq. 0) then
   write(*,*) 'Test passed'
else
   print*,'Errori cnt_err :  ', cnt_err
endif

end
