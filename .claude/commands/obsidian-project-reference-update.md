Project-level override of the global obsidian-project-reference-update skill for the Trading App project.

## Project Context

Before applying any session updates, first read the latest Trading App PROJECT_REFERENCE file to understand current progress and pick up where the last session left off.

**Vault root:**
```
C:\Users\gordo\Google Drive Streaming\.shortcut-targets-by-id\1lIIeHMZyWhQmYyhjk4AacJUByZClFbiv\Merri-Obsidian-KB
```

**File naming convention:** `2026-MM-DD-Trading-App_PROJECT_REFERENCE.md`

**To find the latest file:** look for the most recently dated file matching `*Trading-App_PROJECT_REFERENCE*` in the vault root. Read it in full before proceeding.

**Chat history summary:** `2026-05-16-Trading-App_Chat-History-Summary.md` in the vault root contains a per-session summary of all Claude Code sessions. Read this instead of re-parsing raw JSONL files when session context is needed.

**Archive folder:** `<vault_root>\Archive`

## Workflow

1. Locate and read the most recent `*Trading-App_PROJECT_REFERENCE*` file in the vault root
2. Review its contents to understand current project state and progress
3. Apply session updates to the relevant sections
4. Write a new file named `2026-MM-DD-Trading-App_PROJECT_REFERENCE.md` with today's date
5. Move the previous dated file to the Archive folder with an incrementing version suffix (e.g., `_v1`, `_v2`)

## Edge Cases

**File already has today's date:**
Skip the rename step — write updates to the existing today-dated file. Still archive the previous dated file if it differs from today.

**Archive folder doesn't exist:**
Create it silently before moving.

**No existing reference file found:**
Ask the user to confirm the vault root path before proceeding. Do not create a new file from scratch without explicit instruction.

**User provides session notes inline:**
Incorporate them directly into the relevant sections rather than appending as raw text.

**User says "just update the date and archive":**
Skip content updates — only update the `updated` frontmatter field and session summary date, then rename and archive.
