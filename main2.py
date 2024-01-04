import random
import multiprocessing
import time

# Task 1: Matrix Multiplication Function
def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    # Check if the matrices can be multiplied
    assert cols_A == rows_B, "Matrix dimensions are not compatible for multiplication."

    # Initialize the result matrix
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Perform matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

# Task 2: Serial Implementation
def serial_matrix_multiply(A, B):
    return matrix_multiply(A, B)

# Task 3 and 4: Data-Level Parallelism
def parallel_matrix_multiply(A, B, num_processes):
    # Initialize the result matrix
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    # Define block size
    block_size = len(A) // num_processes

    # Function to perform matrix multiplication on a block
    def multiply_block(start_row, end_row, result_queue):
        local_result = [[0 for _ in range(len(B[0]))] for _ in range(end_row - start_row)]
        for i in range(start_row, end_row):
            for j in range(len(B[0])):
                for k in range(len(A[0])):
                    local_result[i - start_row][j] += A[i][k] * B[k][j]
        result_queue.put(local_result)

    # Create and start processes
    processes = []
    result_queue = multiprocessing.Queue()
    for i in range(num_processes):
        start_row = i * block_size
        end_row = (i + 1) * block_size if i < num_processes - 1 else len(A)
        process = multiprocessing.Process(target=multiply_block, args=(start_row, end_row, result_queue))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Collect results from the queue
    for _ in range(num_processes):
        local_result = result_queue.get()
        for i in range(len(local_result)):
            result[i + start_row] = local_result[i]

    return result

# Task 5: Performance Comparison
def measure_performance(A, B, num_processes):
    start_time = time.time()
    serial_result = serial_matrix_multiply(A, B)
    serial_time = time.time() - start_time

    start_time = time.time()
    parallel_result = parallel_matrix_multiply(A, B, num_processes)
    parallel_time = time.time() - start_time

    speedup = serial_time / parallel_time
    efficiency = speedup / num_processes

    print(f"Serial Execution Time: {serial_time:.6f} seconds")
    print(f"Parallel Execution Time ({num_processes} processes): {parallel_time:.6f} seconds")
    print(f"Speedup: {speedup:.2f}")
    print(f"Efficiency: {efficiency:.2%}")

# Define the dimensions of the matrices
rows_A, cols_A = 30, 12
rows_B, cols_B = 12, 30

# Generate random matrices A and B
A = [[random.randint(1, 10) for _ in range(cols_A)] for _ in range(rows_A)]
B = [[random.randint(1, 10) for _ in range(cols_B)] for _ in range(rows_B)]

# Number of processes for parallel processing
num_processes = 2  # Adjust as needed

# Measure and compare performance
measure_performance(A, B, num_processes)
