import multiprocessing as mp
from typing import TypeVar, Sequence
import logging

T = TypeVar('T')


def run_data_job_in_parallel(data: Sequence[T],
                             job,
                             num_of_instances: int = mp.cpu_count()) -> Sequence[T]:
    # The length of a batch which each process will work on
    batch_size = len(data) // num_of_instances
    # The rest of the data segment if there are any, for the last process
    mod = len(data) % num_of_instances

    batches = []
    start = 0
    for i in range(num_of_instances):
        # If this is the last thread it might have to work on more data compare to the others
        end = start + batch_size + (1 if i < mod else 0)
        batch = data[start:end]
        batches.append((batch, start))
        start = end

    # Infer data in parallel
    with mp.Pool(processes=num_of_instances) as pool:
        results = list(pool.imap(job, batches))

    # Sort inference results by original index
    results.sort(key=lambda x: x[0])

    return results


# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
