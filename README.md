# Analysis of Serial and Parallel Matrix Multiplication

Task 1: This task was pretty straightforward forward I simply implemented a matrix multiplication function that multiplies two matrixes, using Python.

Task 2: Created a matrix serial multiplication function that uses O(n^3) time to do the matrix multiplication of two matrixes. This makes the baseline for further optimization to be compared with. 

Task 3: To further optimize the matrix multiplication, first of all, I determined the block size considering the number of processes that I wanted to run then divided the data into separate blocks and passed them as chunks on each process to handle each multiplication parallelly and combined each process result on a single process queue and finally join all the processes results on a single local result, it is done to make sure that parallel processes don’t execute discriminately and the matrix multiplication results are not wrongly indexed.

To evaluate the result I stored the execution start time and found out the serial matrix multiplication execution and parallel matrix multiplication execution time, also I calculated the amount of speed up and efficiency gained through parallel execution considering the number of processes used.

I’m running a loop to produce matrixes of different dimensions to show how parallel matrix multiplication performs better when the matrix dimension increases because for small matrixes the overhead associated with creating and managing processes will be significant and serial matrix multiplication will result in more efficient but as the dimension increases parallel matrix multiplication gradually performs better. I plot this result on a graph that clearly shows the overall efficiency of parallel matrix multiplication with serial matrix multiplication.

Task 4: However, in terms of multithreading we are creating multiple threads to handle the processing of each block but each thread is executing once but in parallel because, in Python, the Global Interpreter Lock (GIL) restricts the execution of multiple threads in the same process to one at a time. As a result, our CPU-bound task matrix multiplication using multithreading is not producing as efficient result as matrix multiplication using multiprocessing. This is shown in graphs.  










