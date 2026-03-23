#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline runner for managing pipeline task execution.

This module provides the PipelineRunner class that handles the execution
of pipeline tasks with signal handling, garbage collection, and lifecycle
management.
"""

import asyncio
import gc
import signal
from typing import Optional

from loguru import logger

from pipecat.pipeline.base_task import PipelineTaskParams
from pipecat.pipeline.task import PipelineTask
from pipecat.utils.asyncio import compat
from pipecat.utils.base_object import BaseObject


class PipelineRunner(BaseObject):
    """Manages the execution of pipeline tasks with lifecycle and signal handling.

    Provides a high-level interface for running pipeline tasks with automatic
    signal handling (SIGINT/SIGTERM), optional garbage collection, and proper
    cleanup of resources.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        handle_sigint: bool = True,
        handle_sigterm: bool = False,
        force_gc: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """Initialize the pipeline runner.

        Args:
            name: Optional name for the runner instance.
            handle_sigint: Whether to automatically handle SIGINT signals.
            handle_sigterm: Whether to automatically handle SIGTERM signals.
            force_gc: Whether to force garbage collection after task completion.
            loop: Event loop to use. If None, uses the current running loop.
        """
        super().__init__(name=name)

        self._tasks = {}
        self._sig_task = None
        self._force_gc = force_gc
        # TODO(anyio): asyncio.AbstractEventLoop has no trio equivalent. The
        # loop is passed through to PipelineTaskParams for legacy asyncio
        # callers; under trio this stays None and downstream code must cope.
        if loop is not None:
            self._loop = loop
        elif compat.current_backend() == "asyncio":
            self._loop = asyncio.get_running_loop()
        else:
            self._loop = None

        if handle_sigint:
            self._setup_sigint()

        if handle_sigterm:
            self._setup_sigterm()

    async def run(self, task: PipelineTask):
        """Run a pipeline task to completion.

        Args:
            task: The pipeline task to execute.
        """
        logger.debug(f"Runner {self} started running {task}")
        self._tasks[task.name] = task

        # PipelineTask handles cancellation to shutdown the pipeline
        # properly and re-raises it in case there's more cleanup to do.
        try:
            params = PipelineTaskParams(loop=self._loop)
            await task.run(params)
        except compat.get_cancelled_exc_class():
            pass

        del self._tasks[task.name]

        # Cleanup base object.
        await self.cleanup()

        # If we are cancelling through a signal, make sure we wait for it so
        # everything gets cleaned up nicely.
        if self._sig_task:
            await self._sig_task

        if self._force_gc:
            self._gc_collect()

        logger.debug(f"Runner {self} finished running {task}")

    async def stop_when_done(self):
        """Schedule all running tasks to stop when their current processing is complete."""
        logger.debug(f"Runner {self} scheduled to stop when all tasks are done")
        await compat.gather(*[t.stop_when_done() for t in self._tasks.values()])

    async def cancel(self):
        """Cancel all running tasks immediately."""
        logger.debug(f"Cancelling runner {self}")
        await self._cancel()

    async def _cancel(self):
        """Cancel all running tasks immediately."""
        await compat.gather(*[t.cancel() for t in self._tasks.values()])

    def _setup_sigint(self):
        """Set up signal handlers for graceful shutdown."""
        # TODO(anyio): loop.add_signal_handler is asyncio-only. Trio/anyio
        # exposes signals via anyio.open_signal_receiver (an async iterator)
        # which requires a running task rather than a sync callback. Under
        # trio, signal handling must be restructured to run inside run().
        if compat.current_backend() != "asyncio":
            logger.warning(
                "PipelineRunner signal handling is not yet supported under "
                f"{compat.current_backend()}; SIGINT will not be caught."
            )
            return
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, lambda *args: self._sig_handler())
        except NotImplementedError:
            # Windows fallback
            signal.signal(signal.SIGINT, lambda s, f: self._sig_handler())

    def _setup_sigterm(self):
        """Set up signal handlers for graceful shutdown."""
        # TODO(anyio): see _setup_sigint for the trio limitation.
        if compat.current_backend() != "asyncio":
            logger.warning(
                "PipelineRunner signal handling is not yet supported under "
                f"{compat.current_backend()}; SIGTERM will not be caught."
            )
            return
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, lambda *args: self._sig_handler())
        except NotImplementedError:
            # Windows fallback
            signal.signal(signal.SIGTERM, lambda s, f: self._sig_handler())

    def _sig_handler(self):
        """Handle interrupt signals by cancelling all tasks."""
        # TODO(anyio): asyncio.create_task from a sync callback has no trio
        # equivalent (trio requires a nursery). This path is only reachable
        # under asyncio because _setup_sigint/_setup_sigterm bail out early
        # on other backends.
        if not self._sig_task:
            self._sig_task = asyncio.create_task(self._sig_cancel())

    async def _sig_cancel(self):
        """Cancel all running tasks due to signal interruption."""
        logger.warning(f"Interruption detected. Cancelling runner {self}")
        await self.cancel()

    def _gc_collect(self):
        """Force garbage collection and log results."""
        collected = gc.collect()
        logger.debug(f"Garbage collector: collected {collected} objects.")
        logger.debug(f"Garbage collector: uncollectable objects {gc.garbage}")
