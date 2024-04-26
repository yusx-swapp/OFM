import torch
import torch.distributed as dist
import os
import time
from torch.multiprocessing import Process, Manager


def worker(rank, size, shared_queue, sum_storage):
    """Function to be executed by each process."""
    print(f"Rank {rank} of {size} starting.")

    # Simulated training loop
    for epoch in range(10):  # Reduced loop count for simplicity

        local_tensor = rank * 10 + epoch

        if sum_storage[0] != 0:
            local_tensor = sum_storage[0]

        # Append the local tensor to the shared queue
        shared_queue.put(local_tensor)

        # Wait for all processes to reach this point
        dist.barrier()

        if rank == 0:
            # Only the rank 0 prints the size to avoid clutter
            print(f"Epoch {epoch}, shared_queue size: {shared_queue.qsize()}")

        print(f"Rank {rank} updated local_tensor with broadcasted sum: {local_tensor}")

        # Ensure all ranks reach this point before proceeding to the next epoch
        dist.barrier()


def queue_monitor(shared_queue, sum_storage):
    """Monitor and manage the queue size."""
    while True:
        if shared_queue.qsize() > 10:
            sum_val = 0
            for _ in range(5):
                if not shared_queue.empty():
                    sum_val += shared_queue.get()
            monitor_dics = {
                "sum_val": sum_val,
                "queue_size": shared_queue.qsize(),
            }
            sum_storage[0] = (
                sum_val  # Store the sum in the shared list to be accessed by worker processes
            )
            print(f"Calculated sum of popped elements: {sum_val}")
        # time.sleep(1)  # Sleep to prevent this loop from consuming too much CPU


def init_process(rank, size, shared_queue, sum_storage, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)

    worker(rank, size, shared_queue, sum_storage)


if __name__ == "__main__":
    # Number of processes
    size = 4
    processes = []

    # Create a manager for holding the shared queue and sum storage
    manager = Manager()
    shared_queue = manager.Queue()
    sum_storage = manager.list([0])  # Initialize sum storage

    # Start the monitoring process
    monitor_process = Process(target=queue_monitor, args=(shared_queue, sum_storage))
    monitor_process.start()

    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, shared_queue, sum_storage))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # monitor_process.join()

    monitor_process.terminate()  # Terminate the monitor process after all workers have finished
    print("Processing complete.")
