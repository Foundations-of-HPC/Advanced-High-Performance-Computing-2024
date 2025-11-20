! start the module containing the matmul kernel

#define NB 32

module mmul_mod

   use cudafor
   
   contains
   
   ! mmul_shared_kernel computes A*B into C using shared memory
   attributes(global) subroutine mmul_shared_kernel( A, B, C, N )
      ! This thread computes C(i,j) = sum(A(i,:) * B(:,j))
      integer, value :: N
      real(kind=4) :: A(N,N), B(N,N), C(N,N)
      integer :: it, jt
      integer :: i_offset, j_offset, k_offset
      integer :: k_block
      ! submatrices stored in shared memory
      real(kind=4), shared :: As(NB,NB), Bs(NB,NB)
      ! the value of C(i,j) being computed
      real(kind=4) :: Cvalue

      ! Get the offset for i and j indices
      i_offset = (blockidx%x -1) * NB
      j_offset = (blockidx%y -1) * NB

      ! Get the thread indices
      it = threadidx%x
      jt = threadidx%y

      ! Each thread computes one element of Csub   
      ! by accumulating results into Cvalue
      Cvalue = 0.0

      ! Do the loop in chunks of NB, the block size
      do k_block = 1, N/NB

         ! Get the k-th block index
         k_offset = (k_block-1)*NB

         ! Fill the submatrices
         ! Each of the NBxNB threads in the thread block
         ! loads one element of As and Bs
         ! ----------------------- !
         ! INSERT CUDAFORTRAN CODE !
         ! ----------------------- !


         ! Wait until all elements are filled
         ! ----------------------- !
         ! INSERT CUDAFORTRAN CODE !
         ! ----------------------- !


         ! Multiply the two submatrices
         ! Each of the NBxNB threads accumulates the
         ! dot product for its element of C(i,j)
         do k = 1, NB
           ! ----------------------- !
           ! INSERT CUDAFORTRAN CODE !
           ! ----------------------- !

         enddo

         ! Synchronize to make sure all threads are done
         ! reading the submatrices before overwriting them
         ! in the next iteration of the kb loop
         ! ----------------------- !
         ! INSERT CUDAFORTRAN CODE !
         ! ----------------------- !

      enddo
      ! Each of the NBxNB threads stores its element
      ! to the global C array
      ! ----------------------- !
      ! INSERT CUDAFORTRAN CODE !
      ! ----------------------- !


   end subroutine mmul_shared_kernel
   
   
   ! The host routine to drive the matrix multiplication
   subroutine mmul_gpu(h_A, h_B, h_C, N)
      use cudafor
      integer N,ierr
      real(kind=4), dimension(N,N) :: h_A, h_B, h_C
      real(kind=4), dimension(:,:), allocatable, device :: d_A, d_B, d_C
      type(dim3) :: dimGrid, dimBlock
      type(cudaEvent) :: start_event,end_event
      real :: elapsed_cutime
      
      ierr = cudaEventCreate(start_event)
      ierr = cudaEventCreate(end_event)
      
      ! Allocate matrices on the device
      allocate( d_A(N,N), d_B(N,N), d_C(N,N) )
   
      ! Copy matrices A and B in the device
      d_A = h_A
      d_B = h_B
   
      ! Grid specify
      dimBlock = dim3(NB, NB, 1)
      dimGrid  = dim3(int(N/dimBlock%x), int(N/dimBlock%y), 1)
      
      ! Start timing
      ierr = cudaEventRecord(start_event,0)
      
      ! Invoke kernel
      call mmul_shared_kernel <<<dimGrid,dimBlock>>>(d_A, d_B, d_C, N)
      
      ! End timing
      ierr = cudaEventRecord(end_event,0)
      ierr = cudaEventSynchronize(end_event)
      ierr = cudaEventElapsedTime(elapsed_cutime,start_event,end_event)
      
      mflops = 2.0*float(N)*float(N)*float(N)/(1000.0*1000.0)
      print*,'Dimension = ', N
      print*,'CUDA Event elapsed time (sec) = ', elapsed_cutime/1000.0
      print*,'CUDA Gflops                   = ', &
                          mflops/(elapsed_cutime/1000.)/1000.0
      
      ! Copy the result in the host
      h_C = d_C
   
      ! Free the device
      deallocate( d_A, d_B, d_C )
   
   end subroutine mmul_gpu

end module mmul_mod


subroutine mmul_cpu( A, B, C, N )
   integer N
   real(kind=4), dimension(N,N) :: A, B, C
   integer i,j,k
   C = 0.
   do j=1,N
     do k=1,N
        do i=1,N
           C(i,j) = C(i,j) + A(i,k)*B(k,j)
        enddo
     enddo
   enddo
end subroutine mmul_cpu


program main_mmul

   use mmul_mod
   
   ! Dimension of the matrices (N x N)
   integer, parameter :: N = 32*NB

   ! Matrices on the host
   real(kind=4), dimension(:,:), allocatable :: h_A, h_B
   ! Results on the host
   real(kind=4), dimension(:,:), allocatable :: cpu_result, gpu_result

   integer :: i,j
   integer :: cnt_err
   
   ! Allocate matrices on the host
   allocate(h_A(N, N), h_B(N, N))
   allocate(gpu_result(N, N), cpu_result(N, N))
   
   ! Init matrices A and B
   do i = 1, N
      do j = 1, N
         call RANDOM_NUMBER(x)
         h_A(i,j) = x
         call RANDOM_NUMBER(x)
         h_B(i,j) = x
      enddo
   enddo
   
   ! Compute on GPU
   call mmul_gpu (h_A, h_B, gpu_result, N)
   
   ! Compute on CPU
   call mmul_cpu (h_A, h_B, cpu_result, N)


   ! Check results
   cnt_err = 0
   errore = 0.
   do j = 1, N
      do i = 1, N
         errore = max(abs(cpu_result(i,j)-gpu_result(i,j)),errore)
         if(errore.gt.1e-4) cnt_err = cnt_err + 1
      enddo
   enddo
   if (cnt_err .eq. 0) then
      write(*,*) 'Test passed'
   else
      print*,'Errori cnt_err :  ', cnt_err
   endif

end
