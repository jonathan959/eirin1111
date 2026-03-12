# Single-branch workflow (main only)

Use **one branch (main)** so that:
- The web/cloud agent works on **main**
- You get updates by **pulling main** on your PC
- You send updates by **pushing main** from your PC

---

## 1. One-time setup: merge cursor branch into main, then delete it

Do this once so that **main** has everything and the extra branch is gone.

### Option A: GitHub Desktop

1. **Fetch** (get latest from GitHub).
2. Make sure you're on **main**.
3. **Branch → Merge into current branch…**
4. Choose **cursor/development-environment-setup-3a90** (or whatever the agent’s branch is called).
5. Click **Merge**, then **Push origin** so GitHub’s main is updated.
6. **Branch → Delete…** and delete **cursor/development-environment-setup-3a90**.
   - If it asks “also delete from remote”, say **Yes** so the branch is removed from GitHub too.

### Option B: Command line (Git)

From the project folder:

```powershell
$git = "C:\Program Files\Git\cmd\git.exe"
cd "C:\Users\jonat\OneDrive\Desktop\local_3comas_clone_v2"

# Get latest
& $git fetch origin --prune

# Make sure we're on main and it's up to date
& $git checkout main
& $git pull origin main

# Merge the cursor branch into main (so main has everything)
& $git merge origin/cursor/development-environment-setup-3a90 -m "Merge cursor branch into main (single-branch setup)"

# If there are conflicts, fix them, then: git add . ; git commit -m "Resolve merge conflicts"

# Push updated main to GitHub
& $git push origin main

# Delete the cursor branch on GitHub (only main remains)
& $git push origin --delete cursor/development-environment-setup-3a90
```

After this, **main** is the only branch on GitHub.

---

## 2. Point the Cursor cloud agent at main

The cloud agent may be configured to use a specific branch (e.g. the old cursor branch). So that everything goes into **main**:

1. In **Cursor**, open the Cloud Agents / Onboard settings (where you added secrets).
2. Look for a setting like **“Branch”**, **“Working branch”**, or **“Repository branch”**.
3. Set it to **main** (and save).

If you don’t see a branch setting, the agent might already use the default branch (main); in that case you don’t need to change anything.

---

## 3. Your day-to-day workflow

| What you want | What you do |
|---------------|-------------|
| Get the latest (including agent changes) | **Pull** in GitHub Desktop (or `git pull origin main`) |
| Send your (or Cursor’s) changes to GitHub | **Push** in GitHub Desktop (or `git push origin main`) |

- You only use **main**.
- **Pull** = GitHub → your PC.  
- **Push** = your PC → GitHub (and the agent sees it there).

---

## 4. Summary

- **One branch:** main.
- **Agent** works on main (after you set its branch to main, if needed).
- **You:** Pull to get updates, push to send updates. No other branches to think about.
