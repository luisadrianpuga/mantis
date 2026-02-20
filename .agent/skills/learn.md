## Self-improvement via outcome learning

You have a record of every command you've run and whether it succeeded.
Use this to improve your skill files.

### How to review outcomes
COMMAND: echo "outcome review triggered"
(The system will surface recent_outcome_log in context automatically.)

### How to identify a broken skill pattern
Look for commands with outcome=fail or outcome=empty that appear repeatedly.
Cross-reference with the skill file that prompted them.

### How to fix a skill file
1. READ: .agent/skills/<skillname>.md
2. Identify the failing COMMAND: pattern
3. Replace or annotate it with a working alternative
4. WRITE: .agent/skills/<skillname>.md with the updated content

### Updating news.md after a bad fetch
If curl commands for RSS are returning empty, try:
- Different feed URL for the same source
- grep -oP instead of awk for simpler extraction
- Adding -A 'Mozilla/5.0' to the curl command

### Golden rule
Never remove a working command. Add better alternatives above broken ones.
Mark broken patterns with a comment: # deprecated - returned empty YYYY-MM-DD
