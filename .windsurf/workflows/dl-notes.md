---
description: Add learning notes to Obsidian dl-system vault
---

# Add DL-System Learning Notes to Obsidian

This workflow helps add learning records and insights from the dl-system course to the Obsidian vault at `/Users/roc.han/yetihan/obsidian/dl-system`.

## Steps

1. Ask the user what topic/concept to document (e.g., "Conv backward", "RNNCell", "BatchNorm pitfalls").

2. Determine the appropriate subfolder based on the topic category:
   - `hw1/` ~ `hw4/` — homework-specific notes
   - `concepts/` — general deep learning concepts (e.g., im2col, backprop through time)
   - `debugging/` — debugging experiences and lessons learned
   - `tips/` — practical tips (e.g., numpy version issues, os.listdir ordering)

3. Create a new `.md` file in the target subfolder using the format below. The filename should be kebab-case, e.g., `rnn-cell-implementation.md`.

4. File format (Obsidian-compatible Markdown):
   ```markdown
   ---
   date: YYYY-MM-DD
   tags:
     - dl-system
     - <hw-number or category>
     - <relevant-tags>
   ---

   # <Title>

   ## Summary
   <One-paragraph summary of the concept/lesson>

   ## Details
   <Detailed explanation, code snippets, diagrams if needed>

   ## Key Takeaways
   - <Bullet points of important lessons>

   ## Related
   - [[link-to-related-notes]]
   ```

5. The file should be written to the Obsidian vault path: `/Users/roc.han/yetihan/obsidian/dl-system/<subfolder>/<filename>.md`

**Note:** This path is on the user's local Mac, not the remote server. If running in a remote SSH session, remind the user to sync the file locally or use a different method to transfer the content.
