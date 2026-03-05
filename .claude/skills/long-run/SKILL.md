---
name: run-long
description: Run a long-running command or task in background and monitor it with minimal-thinking polls
---

Enter long-running execution mode for the following task:

$ARGUMENTS

If `$ARGUMENTS` is empty or is natural language (not a shell command), wait for the user's next instruction and execute it in this long-running mode.

## Protocol

1. If the task involves a shell command: start it with `run_in_background: true` in the Bash tool call
2. Poll with `TaskOutput(task_id, block=true, timeout=60000)` — at most once per minute
   - Track previous output length; inspect only newly added lines each poll
3. **During each poll: minimal thinking only.** Scan new lines for obvious errors:
   - traceback / exception
   - CUDA OOM
   - NaN loss
   - assertion failed
   - connection refused / timeout
   One-line mental note is enough — do not analyze deeply.
4. **Resume full thinking only when:**
   - Error/anomaly detected → decide whether to `TaskStop`, then explain to user
   - Script finishes or fails → summarize results to user
5. If the task is natural language (e.g., "help me refactor X"), execute it step by step with full autonomy until done, then report back.
6. Repeat until task completes or is stopped
