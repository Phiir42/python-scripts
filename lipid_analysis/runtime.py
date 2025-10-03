# lipid_analysis/runtime.py
import io
import traceback
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from typing import Optional

import matplotlib.pyplot as plt


@contextmanager
def capture_logs_on_failure(label: str, enabled: bool = True):
    """
    Swallow stdout/stderr and suppress plt.show() while running a block.
    If an exception occurs, print the traceback + everything that was buffered.

    Notes:
    - We restore Matplotlib's prior interactive state.
    - We print AFTER the redirects are removed so output is visible.
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

    def _noop_show(*_args, **_kwargs):
        return

    # Enter redirection
    with ExitStack() as stack:
        # Disable interactive draw/GUI popups during capture
        plt.ioff()
        if orig_show is not None:
            plt.show = _noop_show  # type: ignore[assignment]

        stack.enter_context(redirect_stdout(buf))
        stack.enter_context(redirect_stderr(buf))

        try:
            yield
        except Exception as e:
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

    # Now that redirects are OFF, print the captured report (if any), then re-raise
    if err is not None:
        print(f"\n[FAIL] {label}: {err}")
        if tb_text:
            print(tb_text)
        print(f"\n--- Captured log for {label} ---")
        if captured:
            print(captured)
        raise err
