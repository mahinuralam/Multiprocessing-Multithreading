import random
import threading
import time
import matplotlib.pyplot as plt


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
def parallel_matrix_multiply(A, B, num_threads):
    # Initialize the result matrix
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    # Define block size
    block_size = len(A) // num_threads

    # Function to perform matrix multiplication on a block
    def multiply_block(start_row, end_row):
        nonlocal result
        for i in range(start_row, end_row):
            for j in range(len(B[0])):
                for k in range(len(A[0])):
                    result[i][j] += A[i][k] * B[k][j]

    # Create and start threads
    threads = []
    for i in range(num_threads):
        start_row = i * block_size
        end_row = (i + 1) * block_size if i < num_threads - 1 else len(A)
        thread = threading.Thread(target=multiply_block, args=(start_row, end_row))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return result


# Task 5: Performance Comparison
def measure_performance(A, B, num_threads):
    start_time = time.time()
    serial_result = serial_matrix_multiply(A, B)
    serial_time = time.time() - start_time

    start_time = time.time()
    parallel_result = parallel_matrix_multiply(A, B, num_threads)
    parallel_time = time.time() - start_time

    # Check if the results are the same
    assert serial_result == parallel_result, "Results do not match!"
    
    speedup = serial_time / parallel_time
    efficiency = speedup / num_threads

    print(f"Serial Execution Time: {serial_time:.6f} seconds")
    print(f"Parallel Execution Time ({num_threads} threads): {parallel_time:.6f} seconds")
    print(f"Speedup: {speedup:.2f}")
    print(f"Efficiency: {efficiency:.2%}")
    
    return serial_time, parallel_time, speedup, efficiency


def plot_data(size, serial_time, parallel_time, speedup, efficiency):
    plt.figure(figsize=(10, 6))
    plt.plot(size, serial_time, label='Serial Time')
    plt.plot(size, parallel_time, label=f'Parallel Time ({num_threads} threads)')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    sizes = list(range(1, 100))
    num_threads = 4  # Adjust as needed

    serial_times = []
    parallel_times = []
    speedups = []
    efficiencies = []

    for size in sizes:
        rows_A, cols_A = size, size
        rows_B, cols_B = size, size
        A = [[random.randint(1, 10) for _ in range(cols_A)] for _ in range(rows_A)]
        B = [[random.randint(1, 10) for _ in range(cols_B)] for _ in range(rows_B)]

        print(f"For {size}, {size} * {size}, {size} matrix:")
        serial_time, parallel_time, speedup, efficiency = measure_performance(A, B, num_threads)

        serial_times.append(serial_time)
        parallel_times.append(parallel_time)
        speedups.append(speedup)
        efficiencies.append(efficiency)

    # After measuring performance for all sizes, plot the results
    plot_data(sizes, serial_times, parallel_times, speedups, efficiencies)
