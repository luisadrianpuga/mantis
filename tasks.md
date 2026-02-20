# Mantis Tasks

- name: morning briefing
  schedule: daily 8am
  prompt: Run the now command, check todo_list.txt if it exists, and give a brief morning summary.

- name: news digest
  schedule: daily 9am
  prompt: Fetch top headlines by running: curl -sL https://feeds.apnews.com/rss/apf-topnews | grep -oP '(?<=<title>)[^<]+' | head -6

- name: disk check
  schedule: daily 11pm
  prompt: Check disk usage with df -h. If any partition is over 80% full, alert the user clearly.

- name: soul review
  schedule: weekly sunday 10am
  prompt: Review this week's memory. What did you learn about the user's preferences, working style, or goals? Propose 1-3 short additions to SOUL.md as a numbered list. Do not write anything yet - wait for the user to say 'approve soul'.

- name: learn review
  schedule: weekly sunday 11am
  prompt: You are reviewing your own command execution history to improve your skills.
    Step 1 - Run: COMMAND: echo "reviewing outcomes"
    Step 2 - The system will show you recent_outcome_log in context. Analyze it.
    Step 3 - For each skill that had failures or empty results, read the skill file.
    Step 4 - Rewrite the skill file with WRITE: to fix the failing patterns.
    Step 5 - Write one line to MEMORY.md summarizing what changed and why.
    Focus on news.md first - it has the most execution history.
    Only update skills where you have clear evidence of what works better.
    Do not remove working commands. Add alternatives above failing ones.
