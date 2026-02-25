
---
description: Deploy the bot to the server immediately without asking for permission.
---
// turbo-all

**Rule:** After changing app code (Python, templates, static), always run deploy so the **live site** (http://18.224.60.44) is updated. Local edits alone do not update the live site.

1. Run the deployment script from the project root:
```powershell
powershell -ExecutionPolicy Bypass -File deploy.ps1
```
Or: `.\deploy.ps1`

2. The script copies files to EC2, restarts `tradingserver`, and runs a health check. Ensure it completes successfully.
