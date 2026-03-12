# AWS Server Project — Local project (your files, not GitHub) → deploys to AWS

**Server URL:** http://3.148.6.246/  
**Direct app:** http://3.148.6.246:8000

This folder is your **local project** (on your Desktop: `local_3comas_clone_v2`). All changes are made here. This is the project you upload and deploy to the AWS server above.

---

## One script: upload local project and deploy on AWS

From your **local machine** (PowerShell, in this folder):

**First-time setup (once):**
```powershell
$env:AWS_DEPLOY_USER = "ec2-user"    # or "ubuntu" for Ubuntu AMI
$env:AWS_DEPLOY_KEY = "C:\path\to\your-key.pem"
```

**Then run:**
```powershell
.\UPLOAD_AND_DEPLOY_TO_AWS.ps1
```

This uploads this entire folder to the server and runs the deploy there. Your app will be at http://3.148.6.246:8000 (or through your proxy at http://3.148.6.246/).

---

## Deploy files in this project (all in this folder)

| File | Use |
|------|-----|
| **UPLOAD_AND_DEPLOY_TO_AWS.ps1** | Run from your PC: uploads this project to 3.148.6.246 and deploys (needs SSH key). |
| **deploy.ps1** | Main deploy script. Local by default; use with `DEPLOY_AWS=1` on the server. |
| **deploy_aws.ps1** | AWS deploy on Windows. Run **on EC2**: `.\deploy_aws.ps1` |
| **deploy_aws.sh** | AWS deploy on Linux. Run **on EC2**: `./deploy_aws.sh` |
| **START_SERVER.bat** | Local only: double-click to start on port 8000. |
| **START_SERVER_8001.bat** | Local only: double-click to start on port 8001. |

---

## Manual deploy (if you already copied files to EC2)

1. Copy this **entire project folder** to your EC2 instance (e.g. with the upload script above).
2. On the EC2 instance:
   - **Windows:** `.\deploy_aws.ps1`
   - **Linux:** `./deploy_aws.sh`
3. App listens on `0.0.0.0:8000` → http://3.148.6.246:8000

If you use nginx so that http://3.148.6.246/ (port 80) goes to the app, point the proxy to `127.0.0.1:8000`.
