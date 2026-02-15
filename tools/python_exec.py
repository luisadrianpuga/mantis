import io
import textwrap
from contextlib import redirect_stdout


def run(code: str) -> str:
    """
    Execute arbitrary Python code and return stdout or errors.
    """
    cleaned = textwrap.dedent(code)
    local_vars = {}
    stdout = io.StringIO()
    try:
        with redirect_stdout(stdout):
            exec(cleaned, {}, local_vars)
        output = stdout.getvalue().strip()
        # Surface a common "result" variable if user set it.
        if "result" in local_vars:
            output = f"{output}\nresult = {local_vars['result']}".strip()
        return output or "Execution finished with no output."
    except Exception as exc:  # pragma: no cover - defensive guardrail
        return f"Python execution error: {exc}"
