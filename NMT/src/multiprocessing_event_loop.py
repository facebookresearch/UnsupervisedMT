# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
import os
import signal
import threading
from torch import multiprocessing
import uuid


class MultiprocessingEventLoop(object):
    """Start a multiprocessing event loop."""

    def __init__(self, device_ids=None, multiprocessing_method='spawn'):
        super().__init__()
        self.device_ids = tuple(device_ids)
        self.num_replicas = len(device_ids)
        self.rank = None

        self._mp = multiprocessing.get_context(multiprocessing_method)

        self._start_error_handler()
        self._start_multiprocessing()

    def call_async(self, rank, action, result_type=None, fetch_all=False,
                   **kwargs):
        """Asynchronously call a function in each child process.

        Call a function named `action` on the rank'th process and return
        a Future with the result.

        result_type can be used to indicate groups of results that are
        equivalent, so that a gen() call will return the first available
        result instead of waiting for the specific result that was enqueued.
        This can be combined with fetch_all, which instead returns all available
        results of a given type at each gen.
        """

        if result_type is None:
            # use a unique id
            result_type = uuid.uuid4()

        def simple_result_generator(expected_type, rank):
            """Handle the simple case where we want just one result."""
            if len(self.return_buffer[expected_type]) > 0:
                yield self.return_buffer[expected_type].pop(0)
                return

            while True:
                result_type, result = self.return_pipes[rank].recv()
                if result_type == expected_type:
                    yield result
                    return
                self.return_buffer[result_type].append(result)

        def fetch_all_result_generator(expected_type, rank):
            """Handle the more complicated case where we want all available
            results of a given type."""
            # We'll return any results that are already finished.
            results = list(self.return_buffer[expected_type])  # use list() to copy
            self.return_buffer[expected_type].clear()

            # But also actively look for more results to return.
            while True:
                for rank in range(self.num_replicas):
                    # consume all available results from this replica
                    while self.return_pipes[rank].poll():
                        result_type, result = self.return_pipes[rank].recv()
                        if result_type == expected_type:
                            results.append(result)
                        else:
                            self.return_buffer[result_type].append(result)
                if len(results) > 0:
                    yield results
                    return

        self.input_pipes[rank].send((result_type, action, kwargs))

        if fetch_all:
            return Future(fetch_all_result_generator(result_type, rank))
        else:
            return Future(simple_result_generator(result_type, rank))

    def stop(self, interrupt_children=False):
        """Stop multiprocessing."""
        for rank in range(self.num_replicas):
            self.input_pipes[rank].close()
            self.return_pipes[rank].close()
            if interrupt_children:
                # send KeyboardInterrupt to children
                os.kill(self.procs[rank].pid, signal.SIGINT)
            else:
                self.procs[rank].join()
        self.error_queue.put((None, None))  # poison pill

    def _start_error_handler(self):
        """Error handler to catch exceptions in child processes."""
        # create a thread to listen for errors in the child processes
        self.error_queue = self._mp.SimpleQueue()
        error_thread = threading.Thread(target=self._error_listener,
                                        daemon=True)
        error_thread.start()

        # create signal handler that executes in the main process/thread and
        # handles errors from child processes
        signal.signal(signal.SIGUSR1, self._signal_handler)

    def _error_listener(self):
        """A thread that listens for errors in the child processes.

        Errors are handled in a signal handler in the main thread.
        """
        (rank, original_trace) = self.error_queue.get()
        if rank is None:  # poison pill, return
            return

        # requeue error and switch to main thread for handling the error
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def _signal_handler(self, signal, frame):
        """Signal handler that handles errors from child processes.

        This signal handler executes in the main/process thread.
        """
        self.stop(interrupt_children=True)
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)

    def _start_multiprocessing(self):
        """Create child processes to run async event loop.

        Each process reads input from a Pipe, performs some computation,
        and returns its output to another Pipe.
        """
        # create child processes
        input_pipes = []
        return_pipes = []
        procs = []
        for rank, id in enumerate(self.device_ids):
            recv_input_pipe, send_input_pipe = self._mp.Pipe(duplex=False)
            recv_return_pipe, send_return_pipe = self._mp.Pipe(duplex=False)
            proc = self._mp.Process(
                target=self._process_event_loop,
                args=(rank, id, recv_input_pipe, send_return_pipe),
                daemon=True)
            proc.start()
            input_pipes.append(send_input_pipe)
            return_pipes.append(recv_return_pipe)
            procs.append(proc)
        self.input_pipes = input_pipes
        self.return_pipes = return_pipes
        self.return_buffer = defaultdict(lambda: [])
        self.procs = procs

    def _process_event_loop(self, rank, device_id, input_pipe, return_pipe):
        """Event loop that runs in each child process.

        Event loop:
        - take an action from the input pipe
        - call the corresponding function in this process
        - put the return value in the return pipe

        Any exceptions are put in the error queue.
        """
        self.rank = rank
        try:
            # event loop
            while True:
                uid, action, kwargs = input_pipe.recv()
                action_fn = getattr(self, action)
                return_pipe.send((uid, action_fn(rank, device_id, **kwargs)))
        except EOFError:
            # input pipe was closed, do nothing
            pass
        except KeyboardInterrupt:
            # killed by parent, do nothing
            pass
        except Exception:
            # propagate exception from child to parent process, keeping
            # original traceback
            import traceback
            self.error_queue.put((rank, traceback.format_exc()))
        finally:
            # cleanup pipes
            input_pipe.close()
            return_pipe.close()


class Future(object):
    """A wrapper around a Python generator, with syntactic sugar."""
    def __init__(self, generator):
        self.generator = generator

    def gen(self):
        return next(self.generator)

    @staticmethod
    def gen_list(gens):
        return [g.gen() for g in gens]

    @staticmethod
    def gen_tuple_list(gens):
        list = [g.gen() for g in gens]
        return zip(*list)

    @staticmethod
    def wrap(value):
        def gen_value():
            yield value
        return Future(gen_value())
