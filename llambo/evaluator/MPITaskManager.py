#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import warnings
import time
import uuid
import signal
import functools
from contextlib import contextmanager
import logging
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="the 'buf' argument is deprecated", category=UserWarning)

# Tags for message types
class Tags(Enum):
    TASK = 1
    RESULT = 2
    TERMINATE = 3
    CANCEL = 4
    STATUS = 5

class Status(Enum):
    UNKNOWN = 0
    IDLE = 1
    BUSY = 2
    TERMINATED = 3

class MPIFuture:
    def __init__(self, task_id):
        self.task_id = task_id
        self._result = None
        self._exception = None
        self._done = False
        self._cancelled = False
    
    def done(self):
        return self._done or self._cancelled
    
    def cancelled(self):
        return self._cancelled
    
    def result(self):
        if not self._done:
            raise ValueError("Result not available yet")
        if self._cancelled:
            raise ValueError("Task was cancelled")
        if self._exception:
            raise self._exception
        return self._result
    
    def _set_result(self, result):
        self._result = result
        self._done = True
    
    def _set_exception(self, exception):
        self._exception = exception
        self._done = True
    
    def _set_cancelled(self):
        self._cancelled = True

class MPITaskPackage:
    def __init__(self, task_id, func, args, kwargs):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self.deliver_time = None

class MPIResultPackage:
    def __init__(self, task_id, result=None, exception=None, cancelled=False):
        self.task_id = task_id
        self.result = result
        self.exception = exception
        self.cancelled = cancelled

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MPITaskManager(metaclass=Singleton):
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.hostname = MPI.Get_processor_name()

        self.result_recv_buffer = None

        self.sub_process_worker = False
        
        if self.size < 2:
            raise ValueError("This task manager requires at least 2 MPI processes")
        
        self.is_master = (self.rank == 0)
        self.running_as_worker = False
        
        if self.is_master:
            self.running_as_master = True
            self.task_queue = deque()
            self.active_tasks = {}  
            self.workers_status = {i: Status.UNKNOWN for i in range(1, self.size)}  
            self.futures = {}  
            self.excuting = False
            self.shutdown_requested = False
        
            signal.signal(signal.SIGINT, self._handle_interrupt)
            signal.signal(signal.SIGTERM, self._handle_interrupt)

    def create_result_recv_buffer(self, max_size):
        if not self.is_master:
            return
        
        if self.excuting:
            return

        if max_size <= 0:
            logger.error("Invalid buffer size %s", max_size)
            return

        self.result_recv_buffer = bytearray(max_size) 
    
    def _handle_interrupt(self, signum, frame):
        logger.debug("Master received interrupt signal %s. Initiating shutdown...", signum)
        self.shutdown(wait=False)
  
# Task managenment
    def submit(self, func, *args, **kwargs):
        if not self.is_master:
            logger.warning("Only the master process can submit tasks")
            return
        
        if self.shutdown_requested:
            logger.error("Task manager is shutting down, cannot submit new tasks")
            return

        if self.excuting:
            logger.error("Task manager is already executing tasks, cannot submit new tasks")
            return
            
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task package
        task = MPITaskPackage(task_id, func, args, kwargs)
        future = MPIFuture(task_id)
        self.futures[task_id] = future
        
        # Add task to queue
        self.task_queue.append(task)
        logger.debug('Task %s added to queue. Queue length: %s', task_id, len(self.task_queue))
        
        return future
    
    def wait(self, futures=None, timeout=None):
        if self.shutdown_requested or self.excuting:
            logger.error("Task manager is shutting down or already executing tasks, cannot wait for futures")
            return (set(), set())
        
        self.excuting = True
        done, not_done = self._wait_for_futures(futures, timeout)
        self.excuting = False
        return (done, not_done)

    def _wait_for_futures(self, futures, timeout=None):
        if not self.is_master:
            logger.warning("Only the master process can wait for futures")
            return (set(), set())

        if futures is None:
            futures = list(self.futures.values())
        
        start_time = time.monotonic()
        not_done = set(futures)
        
        while not_done:
            # Check timeout
            if timeout is not None and time.monotonic() - start_time > timeout:
                break
            
            # Process any pending results/tasks
            self._master_process_messages()
            self._assign_pending_tasks()
            
            # Remove done futures from not_done
            done_futures = {f for f in not_done if f.done()}
            not_done -= done_futures
            
            if not not_done:
                break
            
            # Sleep briefly to avoid busy waiting
            time.sleep(0.2)

        done = set(futures) - not_done
        return (done, not_done)

    def as_completed(self, futures, timeout=None):
        futures = list(futures)  
        start_time = time.monotonic()
        not_done = set(futures)

        self.excuting = True
        try:
            while not_done:
                self._master_process_messages()

                done = [f for f in not_done if f.done()]
                not_done -= set(done)

                for future in done:
                    yield future

                if timeout is not None:
                    elapsed_time = time.monotonic() - start_time
                    if elapsed_time >= timeout:
                        raise TimeoutError()

                self._assign_pending_tasks()

                time.sleep(0.01)  
        finally:
            self.excuting = False
            logger.debug("as_completed finished")

    def _process_result(self, worker_rank, result_package):
        task_id = result_package.task_id
        
        if task_id in self.futures:
            future = self.futures[task_id]
            
            if result_package.exception:
                future._set_exception(result_package.exception)
                logger.debug("Task %s failed with exception: %s", task_id, result_package.exception)
            elif result_package.cancelled:
                future._set_cancelled()
                logger.debug("Task %s was cancelled by worker %s", task_id, worker_rank)
            else:
                result = result_package.result
                future._set_result(result)
                logger.debug("Task %s completed successfully", task_id)
    
# Worker management
    def shutdown(self, wait=True, cancel_futures=False, terminate_workers=False):
        if not self.is_master:
            return
        
        self.shutdown_requested = True
        logger.debug("Shutdown requested with wait=%s, cancel_futures=%s, terminate_workers=%s", wait, cancel_futures, terminate_workers)
        
        if cancel_futures:
            # Cancel all pending futures
            for task in list(self.task_queue):
                future = self.futures[task.task_id]
                future._set_cancelled()
            self.task_queue.clear()
            
            # Cancel all running tasks
            for worker_rank, task in list(self.active_tasks.items()):
                task_id = task.task_id
                self.comm.send(task_id, dest=worker_rank, tag=Tags.CANCEL.value)
        
        if wait and (self.task_queue or self.active_tasks):
            logger.debug("Waiting for pending tasks to complete")
            pending_futures = [self.futures[task.task_id] for task in self.task_queue]
            pending_futures.extend([self.futures[task.task_id] for task in self.active_tasks.values()])
            self._wait_for_futures(pending_futures)
        
        if terminate_workers:
            for worker_rank in range(1, self.size):
                self.comm.send(None, dest=worker_rank, tag=Tags.TERMINATE.value)
                logger.debug("Termination signal sent to worker %s", worker_rank)
        
            if wait:
                logger.debug("Waiting for workers to shut down")
                while any(status != Status.TERMINATED for status in self.workers_status.values()):
                    self._master_process_messages()
                    time.sleep(0.2)
                logger.debug("All workers have shut down")
            self.running = False

        self.shutdown_requested = False
        logger.debug("Shutdown completed")
    
    def _master_process_messages(self):
        # Check for results from any worker
        for worker_rank in range(1, self.size):
            if self.comm.Iprobe(source=worker_rank, tag=Tags.RESULT.value):
                status = MPI.Status()
                try:
                    if self.result_recv_buffer is not None:
                        self.comm.recv(self.result_recv_buffer, source=worker_rank, tag=Tags.RESULT.value, status=status)
                        data_size = status.Get_count(MPI.BYTE)
                        result_package = MPI.pickle.loads(self.result_recv_buffer[:data_size])
                    else:
                        result_package = self.comm.recv(source=worker_rank, tag=Tags.RESULT.value, status=status)
                        data_size = status.Get_count(MPI.BYTE)
                    logger.debug("Received result from worker %s with size %s", worker_rank, data_size)
                    self._process_result(worker_rank, result_package)
                except MPI.Exception as e:
                    logger.error("Received failed result from worker %s", worker_rank)
                    logger.error(e)
                    # add the task back to the queue
                    task = self.active_tasks[worker_rank]
                    self.task_queue.append(task)

                if worker_rank in self.active_tasks:
                    del self.active_tasks[worker_rank]
                self.workers_status[worker_rank] = Status.IDLE
            
            # Check for status updates
            if self.comm.Iprobe(source=worker_rank, tag=Tags.STATUS.value):
                status = self.comm.recv(source=worker_rank, tag=Tags.STATUS.value)
                if status == Status.TERMINATED or status == Status.IDLE:
                    if worker_rank in self.active_tasks:
                        logger.error("Worker %s change status to %s while still running task %s", worker_rank, status, self.active_tasks[worker_rank].task_id)
                        del self.active_tasks[worker_rank]
                self.workers_status[worker_rank] = status
                logger.debug("Worker %s status: %s", worker_rank, status) 
        
    
    def _assign_pending_tasks(self):
        idle_workers = [rank for rank, status in self.workers_status.items() if status == Status.IDLE]
        
        while idle_workers and self.task_queue:
            worker_rank = idle_workers.pop(0)
            task = self.task_queue.popleft()
            
            # Send the task to the worker
            logger.debug("Assigning task %s to worker %s", task.task_id, worker_rank)   
            self.comm.send(task, dest=worker_rank, tag=Tags.TASK.value)
            
            # Mark the worker as busy
            self.workers_status[worker_rank] = Status.BUSY
            self.active_tasks[worker_rank] = task
            
            logger.debug("Task %s sent to worker %s", task.task_id, worker_rank)

    def _execute_task(self, task, queue):
        task_id = task.task_id
        func = task.func
        args = task.args
        kwargs = task.kwargs

        try:
            logger.debug("Worker %s executing task %s", self.rank, task_id)
            result_value = func(*args, **kwargs)
            result = MPIResultPackage(task_id, result=result_value)
            logger.debug("Worker %s completed task %s", self.rank, task_id)
        except Exception as e:
            logger.exception("Worker %s encountered error on task %s", self.rank, task_id)
            result = MPIResultPackage(task_id, exception=e)
        
        # Send the result back to the main process
        if queue:
            queue.put(result)
            return None
        else:
            return result

    def worker_process(self):
        logger.debug("Worker %s started on %s", self.rank, self.hostname)
        self.comm.send(Status.IDLE, dest=0, tag=Tags.STATUS.value)
        
        current_task = None
        sub_process = None
        res_queue = None
        self.running_as_worker = True

        # print_time = time.monotonic()
        
        while self.running_as_worker:
            # Check for termination signal
            if self.comm.Iprobe(source=0, tag=Tags.TERMINATE.value):
                self.comm.recv(source=0, tag=Tags.TERMINATE.value)
                logger.debug("Worker %s received termination signal", self.rank)
                self.comm.send(Status.TERMINATED, dest=0, tag=Tags.STATUS.value)
                logger.debug("Worker %s send termination status", self.rank)
                self.running_as_worker = False
                break
            
            # Check for task cancellation if we're running a task
            if current_task and self.comm.Iprobe(source=0, tag=Tags.CANCEL.value):
                cancel_id = self.comm.recv(source=0, tag=Tags.CANCEL.value)
                if cancel_id == current_task.task_id:
                    logger.debug("Worker %s cancelling task %s", self.rank, cancel_id)
                    if sub_process:
                        result = MPIResultPackage(cancel_id, cancelled=True)
                        self.comm.send(result, dest=0, tag=Tags.RESULT.value)
                        current_task = None

                        sub_process.terminate()
                        sub_process.join()
                        sub_process = None
                        res_queue = None
        
            # Check for new task if we're not running one
            if not current_task and self.comm.Iprobe(source=0, tag=Tags.TASK.value):
                logger.debug("Worker %s is receiving task", self.rank)
                req = self.comm.irecv(source=0, tag=Tags.TASK.value)
                # FIXME: This is a hack to avoid deadlock
                time.sleep(0.5)
                current_task = req.wait()
                logger.debug("Worker %s received task %s", self.rank, current_task.task_id)
                
                if self.sub_process_worker:
                    import multiprocessing as mp

                    res_queue = mp.Queue()
                    logger.debug("Worker %s starting sub process for task %s", self.rank, current_task.task_id)

                    process = mp.Process(target=self._execute_task, args=(current_task, res_queue))
                    sub_process = process
                    process.start()
                else:
                    # Execute the task
                    result = self._execute_task(current_task, None)
                    # Send the result back to the master
                    self.comm.send(result, dest=0, tag=Tags.RESULT.value)
                    current_task = None

            if sub_process and res_queue and not res_queue.empty():
                # finish the sub process
                import queue as queue
                try:
                    result = res_queue.get()

                    self.comm.send(result, dest=0, tag=Tags.RESULT.value)
                    current_task = None

                    if sub_process.is_alive():
                        sub_process.terminate()
                        sub_process.join()
                    sub_process = None
                    res_queue = None

                except queue.Empty:
                    pass

            # Sleep briefly to avoid busy waiting
            time.sleep(0.2)

            # current_time = time.monotonic()
            # if current_time - print_time > 2:
            #     logger.debug("Worker %s is idle", self.rank)
            #     print_time = current_time
        
        self.running_as_worker = False
        logger.debug("Worker %s shut down", self.rank)
    
    def start_worker(self):
        if not self.is_master and not self.running_as_worker:
            self.worker_process()

@contextmanager
def start_mpi_task_manager(result_recv_buffer_size=None, use_sub_process_worker=False):
    task_manager = MPITaskManager()
    task_manager.sub_process_worker = use_sub_process_worker
    if task_manager.is_master:
        if task_manager.shutdown_requested or task_manager.excuting:
            raise RuntimeError("Task manager is already executing tasks or shutting down")

        if result_recv_buffer_size is not None and result_recv_buffer_size > 0:
            task_manager.create_result_recv_buffer(result_recv_buffer_size)

        yield task_manager
        
        task_manager.shutdown(wait=True, cancel_futures=True, terminate_workers=True)
    else:
        yield task_manager
        if not task_manager.running_as_worker:
            task_manager.start_worker()

# Example usage
def compute_intensive_task(task_id, value):
    if np.random.rand() < 0.1:
        raise ValueError("Random failure")
    
    # Simulate computation with large array
    size = 10000
    data = np.linspace(0, 10, size)
    result = np.sum(np.sin(data + value))

    time_out = np.random.randint(1, 5)
    time.sleep(time_out)  # Additional work simulation
    
    return result

def master_test_func(task_manager):
    # Submit tasks
    n_tasks = 9
    futures = []

    print('====================')
    print("Submitting tasks for wait...")
    for i in range(n_tasks):
        future = task_manager.submit(compute_intensive_task, f"task_{i}", i)
        futures.append(future)
    
    # Wait for all tasks to complete
    print("Waiting for tasks to complete...")
    done, not_done = task_manager.wait(futures, timeout=1)
    
    # Process results
    for i, future in enumerate(futures):
        try:
            result = future.result()
            print(f"Result from task {i}: {result:.6f}")
        except Exception as e:
            print(f"Task {i} failed: {str(e)}")
    
    # Shutdown the task manager
    task_manager.shutdown(wait=True, cancel_futures=True, terminate_workers=False)

    print('====================')
    print("Submitting tasks for as_completed...")
    futures = []
    for i in range(n_tasks):
        future = task_manager.submit(compute_intensive_task, f"task_{i}", i)
        futures.append(future) 

    try:
        count = 0
        for future in task_manager.as_completed(futures, timeout=100):
            try:
                count += 1
                result = future.result()
                print(f"Result from task {future.task_id}: {result:.6f}")
                if count == 2:
                    print("Cancelling remaining tasks...")
                    break
            except Exception as e:
                print(f"Task {i} failed: {str(e)}")
                print("Cancelling remaining tasks...")
                break
    except TimeoutError:
        print("TimeoutError: No more tasks to process")

    task_manager.shutdown(wait=True, cancel_futures=True)

    print("Submitting new tasks for as_completed...")
    futures = []
    for i in range(n_tasks):
        future = task_manager.submit(compute_intensive_task, f"task_{i}", i)
        futures.append(future) 

    for future in task_manager.as_completed(futures):
        try:
            result = future.result()
            print(f"Result from task {future.task_id}: {result:.6f}")
        except Exception as e:
            print(f"Task {i} failed: {str(e)}")
            print("Cancelling remaining tasks...")
            break


def test_context_func():
    logging.basicConfig(level=logging.DEBUG)
    with start_mpi_task_manager() as task_manager:
        if task_manager.is_master:
            master_test_func(task_manager)

def test():
    logging.basicConfig(level=logging.DEBUG)
    task_manager = MPITaskManager()
    
    if task_manager.is_master:
        master_test_func(task_manager)
        task_manager.shutdown(wait=True, cancel_futures=True, terminate_workers=True)
    else:
        task_manager.start_worker()

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     test()