# lipid_analysis/runtime.py
"""
Runtime helpers for controlled execution and failure reporting.

- capture_logs_on_failure: context manager that temporarily swallows stdout/stderr
  and disables plt.show() so that noisy operations don't emit during batch runs.
  If an exception occurs, it prints a labeled failure header, the full traceback,
  and all buffered output after restoring normal I/O and Matplotlib state.
"""

from __future__ import annotations

import io
import logging
import traceback
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from typing import Iterator, Optional

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@contextmanager
def capture_logs_on_failure(label: str, enabled: bool = True) -> Iterator[None]:
    """
    Swallow stdout/stderr and suppress plt.show() while running a block.
    If an exception occurs, log the traceback and everything buffered.

    Parameters
    ----------
    label : str
        A short label describing the protected run (included in failure logs).
    enabled : bool, default True
        If False, acts as a no-op context manager.

    Notes
    -----
    - Matplotlib's interactive state is preserved and restored.
    - Logs are emitted *after* redirections are removed so they are visible.
    """
    if not enabled:
        yield
        return

    buf = io.StringIO()
    err: Optional[BaseException] = None
    tb_text: Optional[str] = None
    captured: Optional[str] = None

    # Preserve interactive state and show()
    was_interactive = plt.isinteractive()
    orig_show = getattr(plt, "show", None)

    def _noop_show(*_args, **_kwargs) -> None:
        return

    with ExitStack() as stack:
        # Disable interactive draw/GUI popups during capture
        plt.ioff()
        if orig_show is not None:
            plt.show = _noop_show  # type: ignore[assignment]

        # Redirect both stdout and stderr into a buffer
        stack.enter_context(redirect_stdout(buf))
        stack.enter_context(redirect_stderr(buf))

        try:
            yield
        except Exception as e:  # capture, then re-raise after restoring state
            err = e
            tb_text = traceback.format_exc()
            captured = buf.getvalue()
        finally:
            # Restore Matplotlib state
            if orig_show is not None:
                plt.show = orig_show  # type: ignore[assignment]
            if was_interactive:
                plt.ion()
            else:
                plt.ioff()

    # Emit logs after redirection is off
    if err is not None:
        logger.error("[FAIL] %s: %s", label, err)
        if tb_text:
            logger.error(tb_text.rstrip("\n"))
        logger.error("--- Captured log for %s ---", label)
        if captured:
            logger.error(captured.rstrip("\n"))
        raise err
