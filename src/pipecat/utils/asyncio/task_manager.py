#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async task management.

This module provides task management functionality. Includes both abstract base
classes and concrete implementations for managing async tasks with
comprehensive monitoring and cleanup capabilities.

The :class:`TaskManager` supports both ``asyncio`` and ``trio`` backends via
`anyio <https://anyio.readthedocs.io/>`_. Under asyncio it uses free-floating
tasks (the historical behaviour); under trio it requires a task group passed
via :class:`TaskManagerParams` because trio uses structured concurrency.
"""

import asyncio
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Coroutine, Dict, Optional, Sequence

import anyio
import anyio.abc
from loguru import logger

from pipecat.utils.asyncio.compat import current_backend


@dataclass
class TaskManagerParams:
    """Configuration parameters for task manager initialization.

    Parameters:
        loop: The asyncio event loop (asyncio backend only; ignored under trio).
        task_group: An anyio task group for spawning children. Required when
            running under trio; unused under asyncio.
    """

    loop: Optional[asyncio.AbstractEventLoop] = None
    task_group: Optional[anyio.abc.TaskGroup] = None


class BaseTaskManager(ABC):
    """Abstract base class for asyncio task management.

    Provides the interface for creating, monitoring, and managing asyncio tasks.
    """

    @abstractmethod
    def setup(self, params: TaskManagerParams):
        """Initialize the task manager with configuration parameters.

        Args:
            params: Configuration parameters for task management.
        """
        pass

    @abstractmethod
    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop used by this task manager.

        Returns:
            The asyncio event loop instance.
        """
        pass

    @abstractmethod
    def create_task(self, coroutine: Coroutine, name: str) -> asyncio.Task:
        """Creates and schedules a new asyncio Task that runs the given coroutine.

        The task is added to a global set of created tasks.

        Args:
            coroutine: The coroutine to be executed within the task.
            name: The name to assign to the task for identification.

        Returns:
            The created task object.
        """
        pass

    @abstractmethod
    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Cancels the given asyncio Task and awaits its completion with an optional timeout.

        This function removes the task from the set of registered tasks upon
        completion or failure.

        Args:
            task: The task to be cancelled.
            timeout: The optional timeout in seconds to wait for the task to cancel.
        """
        pass

    @abstractmethod
    def current_tasks(self) -> Sequence[asyncio.Task]:
        """Returns the list of currently created/registered tasks.

        Returns:
            Sequence of currently managed asyncio tasks.
        """
        pass


@dataclass
class TaskData:
    """Internal data structure for tracking task metadata.

    Parameters:
        task: The task handle being managed (``asyncio.Task`` or
            :class:`~pipecat.utils.asyncio.anyio_task_manager.TaskHandle`).
    """

    task: Any


class TaskManager(BaseTaskManager):
    """Backend-agnostic task manager.

    Supports both asyncio (free-floating tasks via ``loop.create_task``) and
    trio (structured concurrency via an anyio task group). The backend is
    detected at :meth:`setup` time. Under trio, a task group must be supplied
    in :class:`TaskManagerParams`; under asyncio an event loop is used.
    """

    def __init__(self) -> None:
        """Initialize the task manager with empty task registry."""
        self._tasks: Dict[str, TaskData] = {}
        self._params: Optional[TaskManagerParams] = None
        self._backend: str = "asyncio"

    def setup(self, params: TaskManagerParams):
        """Initialize the task manager with configuration parameters.

        Args:
            params: Configuration parameters for task management.

        Raises:
            RuntimeError: If running under trio without a task group.
        """
        if self._params:
            return
        self._params = params
        self._backend = current_backend()
        if self._backend == "trio" and params.task_group is None:
            raise RuntimeError("TaskManager requires a task_group when running under trio")
        if self._backend == "asyncio" and params.loop is None:
            params.loop = asyncio.get_running_loop()

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop used by this task manager.

        Returns:
            The asyncio event loop instance.

        Raises:
            Exception: If the task manager is not set up.
            RuntimeError: If running under trio (no event loop concept).
        """
        if not self._params:
            raise Exception("TaskManager is not setup: unable to get event loop")
        if self._backend != "asyncio":
            raise RuntimeError(
                "get_event_loop() is not available under trio; "
                "use pipecat.utils.asyncio.compat primitives instead"
            )
        assert self._params.loop is not None
        return self._params.loop

    def create_task(self, coroutine: Coroutine, name: str):
        """Create and schedule a new task running the given coroutine.

        Under asyncio, returns an :class:`asyncio.Task`. Under trio, returns
        a :class:`~pipecat.utils.asyncio.anyio_task_manager.TaskHandle` that
        duck-types the same API subset.

        Args:
            coroutine: The coroutine to be executed within the task.
            name: The name to assign to the task for identification.

        Returns:
            The created task object or handle.

        Raises:
            Exception: If the task manager is not properly set up.
        """
        if not self._params:
            raise Exception("TaskManager is not setup: unable to create task")

        if self._backend == "trio":
            return self._create_task_trio(coroutine, name)
        return self._create_task_asyncio(coroutine, name)

    def _create_task_asyncio(self, coroutine: Coroutine, name: str) -> asyncio.Task:
        async def run_coroutine():
            try:
                return await coroutine
            except asyncio.CancelledError:
                logger.trace(f"{name}: task cancelled")
                raise
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                last = tb[-1]
                logger.error(f"{name} unexpected exception ({last.filename}:{last.lineno}): {e}")

        assert self._params and self._params.loop
        task = self._params.loop.create_task(run_coroutine())
        task.set_name(name)
        task.add_done_callback(self._task_done_handler)
        self._tasks[name] = TaskData(task=task)
        logger.trace(f"{name}: task created")
        return task

    def _create_task_trio(self, coroutine: Coroutine, name: str):
        from pipecat.utils.asyncio.anyio_task_manager import TaskHandle

        assert self._params and self._params.task_group
        handle = TaskHandle(name)

        async def run() -> None:
            cancelled_exc = anyio.get_cancelled_exc_class()
            try:
                with anyio.CancelScope() as scope:
                    handle._cancel_scope = scope
                    if handle._cancel_requested:
                        scope.cancel()
                    try:
                        handle._result = await coroutine
                    except cancelled_exc:
                        logger.trace(f"{name}: task cancelled")
                        raise
                    except Exception as e:
                        handle._exception = e
                        tb = traceback.extract_tb(e.__traceback__)
                        last = tb[-1]
                        logger.error(
                            f"{name} unexpected exception ({last.filename}:{last.lineno}): {e}"
                        )
            finally:
                handle._done.set()
                self._tasks.pop(name, None)

        self._tasks[name] = TaskData(task=handle)
        self._params.task_group.start_soon(run, name=name)
        logger.trace(f"{name}: task created")
        return handle

    async def cancel_task(self, task, timeout: Optional[float] = None):
        """Cancel the given task and await its completion with optional timeout.

        Args:
            task: The task (``asyncio.Task`` or ``TaskHandle``) to cancel.
            timeout: Optional seconds to wait before giving up.
        """
        name = task.get_name()
        task.cancel()
        cancelled_exc = anyio.get_cancelled_exc_class()
        try:
            if timeout:
                with anyio.fail_after(timeout):
                    await task
            else:
                await task
        except TimeoutError:
            logger.warning(f"{name}: timed out waiting for task to cancel")
        except cancelled_exc:
            pass
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            logger.error(
                f"{name} unexpected exception while cancelling task ({last.filename}:{last.lineno}): {e}"
            )
        except BaseException as e:
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            logger.critical(
                f"{name} fatal base exception while cancelling task ({last.filename}:{last.lineno}): {e}"
            )
            raise

    def current_tasks(self) -> Sequence:
        """Return the list of currently created/registered tasks."""
        return [data.task for data in self._tasks.values()]

    def _task_done_handler(self, task: asyncio.Task):
        """Remove a completed asyncio task from the registry."""
        name = task.get_name()
        self._tasks.pop(name, None)
