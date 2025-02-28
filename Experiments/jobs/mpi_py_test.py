import time
import concurrent.futures
import random
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor
from mpi4py.futures import get_comm_workers


def slow_square(n):
    """A deliberately slow function to simulate computation."""
    # comm = get_comm_workers()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"Rank {rank}: Computing square of {n}...")
    time.sleep(random.uniform(0.1, 0.5))  # Simulate varying computation time
    return n * n

def future_test():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    numbers = list(range(4))

    with MPIPoolExecutor() as executor:
        futures = []
        for number in numbers:
            future = executor.submit(slow_square, number)
            futures.append(future)
            print(f"Rank {rank}: Submitted job for {number}")

        print(f"Rank {rank}: Waiting for results...")
        for future in futures:
            try:
                result = future.result()
                print(f"Rank {rank}: Received result: {result}")
            except Exception as e:
                print(f"Rank {rank}: Job raised exception: {e}")


    # Using MPIPoolExecutor.  Context manager is important to ensure shutdown.
    # executor = MPIPoolExecutor(max_workers=size)  #leave one process for job submission
    # # executor = MPICommExecutor(comm=comm, root=0)  #leave one process for job submission
    # if rank == 0:  # Master process submits jobs, worker does nothing here
    # futures = []
    # for number in numbers:
    #     future = executor.submit(slow_square, number)
    #     futures.append(future)
    #     print(f"Rank {rank}: Submitted job for {number}")

    # print(f"Rank {rank}: Waiting for results...")
    # for future in futures:  # this is the submit loop, it does NOT wait for completion
    #     pass  # We don't need to do anything until we iterate with as_completed


    # concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)


    # results = []
    # should_cancel = False
    # for future in concurrent.futures.as_completed(futures): 
    #     try:
    #         result = future.result()
    #         results.append(result)
    #         print(f"Rank {rank}: Received result: {result}")
    #     except Exception as e:
    #         print(f"Rank {rank}: Job raised exception: {e}")
    #         # Optionally handle the exception (e.g., retry the job)
    #     finally:
    #         pass
    # executor.shutdown(wait=True, cancel_futures=should_cancel)
    # print(f"Rank {rank}: All results: {results}")

    #worker process do nothing except listen in a blocked state

import concurrent.futures
import time

# Function to perform a CPU-bound task
def compute_factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def python_future_test():
    numbers = [50000, 60000, 70000, 80000]

    start_time = time.time()

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_factorial, num) for num in numbers]
        for future in concurrent.futures.as_completed(futures):
            print(f"Factorial computed for number: {numbers[futures.index(future)]}")

    end_time = time.time()
    print(f"Computed all factorials in {end_time - start_time:.2f} seconds")

def send_test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print("Rank: ", rank)

    data = [1, 2, 3, 4, 5]

    if rank == 0:
        print("Rank 0 sending: ", data)
        comm.send(data, dest=1, tag=11)
    elif rank == 1: 
        data = comm.recv(source=0, tag=11)
        print("Rank 1 received: ", data)

if __name__ == '__main__':

    # send_test()
    future_test()
    