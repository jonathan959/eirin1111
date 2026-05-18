# Recover unreachable Eirin server (3.151.143.63) then deploy. Optional AWS start if EIP is in us-east-2.
# Requires AWS API credentials (one of):
#   - aws configure (default profile)
#   - $env:AWS_ACCESS_KEY_ID + $env:AWS_SECRET_ACCESS_KEY
#   - Project file aws-deploy.env with AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
#
# Usage:
#   .\scripts\recover_ec2_and_deploy.ps1
#   .\scripts\recover_ec2_and_deploy.ps1 -DeployOnly   # skip EC2 recovery, deploy if SSH works
#   .\scripts\recover_ec2_and_deploy.ps1 -Quick       # pass -Quick to deploy.ps1

param(
  [switch]$DeployOnly,
  [switch]$Quick
)

$ErrorActionPreference = "Stop"
$LocalRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$hostFile = Join-Path $LocalRoot "deploy_host.txt"
if (-not (Test-Path -LiteralPath $hostFile)) { throw "Missing deploy_host.txt" }
$HostIp = (Get-Content -LiteralPath $hostFile -Raw).Trim()
if (-not $HostIp) { throw "deploy_host.txt is empty" }
$Region = "us-east-2"
$User = "ubuntu"
$RemoteDir = "/home/ubuntu/local_3comas_clone_v2"

function Get-DeployKeyPath {
  foreach ($p in @(
    $env:EIRIN_DEPLOY_KEY,
    (Join-Path $LocalRoot "eirin-bot-key.pem"),
    "C:\Users\jonat\OneDrive\Desktop\server\eirin-bot-key.pem"
  )) {
    if ($p -and (Test-Path -LiteralPath $p)) { return (Resolve-Path -LiteralPath $p).Path }
  }
  throw "No eirin-bot-key.pem found for SSH."
}

function Test-TcpPort {
  param([string]$HostName, [int]$Port, [int]$TimeoutMs = 8000)
  try {
    $c = New-Object System.Net.Sockets.TcpClient
    $iar = $c.BeginConnect($HostName, $Port, $null, $null)
    if (-not $iar.AsyncWaitHandle.WaitOne($TimeoutMs)) { $c.Close(); return $false }
    $c.EndConnect($iar)
    $c.Close()
    return $true
  } catch { return $false }
}

function Load-AwsCredentials {
  $credFile = Join-Path $LocalRoot "aws-deploy.env"
  if (Test-Path $credFile) {
    Get-Content $credFile | ForEach-Object {
      if ($_ -match '^\s*([^#=]+)=(.*)$') {
        $k = $matches[1].Trim()
        $v = $matches[2].Trim().Trim('"').Trim("'")
        if ($k -eq "AWS_ACCESS_KEY_ID") { $env:AWS_ACCESS_KEY_ID = $v }
        if ($k -eq "AWS_SECRET_ACCESS_KEY") { $env:AWS_SECRET_ACCESS_KEY = $v }
        if ($k -eq "AWS_DEFAULT_REGION") { $env:AWS_DEFAULT_REGION = $v }
      }
    }
  }
  if (-not $env:AWS_DEFAULT_REGION) { $env:AWS_DEFAULT_REGION = $Region }
}

function Invoke-AwsCli {
  param([string[]]$Args)
  $out = & python -m awscli @Args --region $env:AWS_DEFAULT_REGION --output json 2>&1
  if ($LASTEXITCODE -ne 0) { throw ($out | Out-String) }
  return $out | ConvertFrom-Json
}

function Get-MyPublicIp {
  try {
    return (Invoke-RestMethod -Uri "https://checkip.amazonaws.com" -TimeoutSec 8).Trim()
  } catch {
    return $null
  }
}

function Ensure-Ec2Reachable {
  Load-AwsCredentials
  if (-not $env:AWS_ACCESS_KEY_ID -or -not $env:AWS_SECRET_ACCESS_KEY) {
    $cfg = & python -m awscli configure list 2>&1 | Out-String
    if ($cfg -notmatch "access_key\s+\w") {
      throw @"
Server $HostIp is unreachable on SSH (port 22). Start the VPS and open port 22 from your IP.

If this host is still on AWS EC2, you can also:
  1) Save API keys to aws-deploy.env and re-run this script (auto-start + SG fix), or
  2) AWS Console -> EC2 -> start the instance bound to $HostIp
"@
    }
  }

  Write-Host "Looking up Elastic IP $HostIp in $Region..." -ForegroundColor Cyan
  $addr = Invoke-AwsCli @("ec2", "describe-addresses", "--public-ips", $HostIp)
  $instId = $addr.Addresses[0].InstanceId
  if (-not $instId) {
    throw "No instance associated with $HostIp. Re-associate the Elastic IP in EC2 console."
  }
  Write-Host "Instance: $instId" -ForegroundColor Gray

  $desc = Invoke-AwsCli @("ec2", "describe-instances", "--instance-ids", $instId)
  $inst = $desc.Reservations[0].Instances[0]
  $state = $inst.State.Name
  $sgIds = @($inst.SecurityGroups | ForEach-Object { $_.GroupId })

  if ($state -eq "stopped") {
    Write-Host "Instance is stopped — starting..." -ForegroundColor Yellow
    Invoke-AwsCli @("ec2", "start-instances", "--instance-ids", $instId) | Out-Null
    Write-Host "Waiting for running state (up to 3 min)..." -ForegroundColor Yellow
    & python -m awscli ec2 wait instance-running --instance-ids $instId --region $env:AWS_DEFAULT_REGION
    Start-Sleep -Seconds 15
  } elseif ($state -ne "running") {
    throw "Instance state is '$state' — fix in AWS Console before deploy."
  } else {
    Write-Host "Instance already running." -ForegroundColor Green
  }

  $myIp = Get-MyPublicIp
  if ($myIp -and $sgIds.Count -gt 0) {
    $cidr = "$myIp/32"
    Write-Host "Ensuring security group allows SSH/HTTP from $cidr ..." -ForegroundColor Cyan
    foreach ($sg in $sgIds) {
      foreach ($rule in @(
        @{ IpProtocol = "tcp"; FromPort = 22; ToPort = 22 },
        @{ IpProtocol = "tcp"; FromPort = 80; ToPort = 80 },
        @{ IpProtocol = "tcp"; FromPort = 443; ToPort = 443 },
        @{ IpProtocol = "tcp"; FromPort = 8000; ToPort = 8000 }
      )) {
        try {
          & python -m awscli ec2 authorize-security-group-ingress `
            --group-id $sg --ip-permissions "IpProtocol=$($rule.IpProtocol),FromPort=$($rule.FromPort),ToPort=$($rule.ToPort),IpRanges=[{CidrIp=$cidr,Description=EirinDeployAuto}]" `
            --region $env:AWS_DEFAULT_REGION 2>$null | Out-Null
        } catch { }
      }
    }
  }

  Write-Host "Waiting for SSH on $HostIp ..." -ForegroundColor Yellow
  $key = Get-DeployKeyPath
  $ok = $false
  for ($i = 0; $i -lt 30; $i++) {
    if (Test-TcpPort -HostName $HostIp -Port 22) {
      $sshTest = & ssh -o ConnectTimeout=6 -o BatchMode=yes -o StrictHostKeyChecking=no -i $key "${User}@${HostIp}" "echo ok" 2>&1
      if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    }
    Start-Sleep -Seconds 4
  }
  if (-not $ok) {
    throw "Instance is running but SSH still fails. Check security group inbound (port 22) and key pair."
  }
  Write-Host "SSH is up." -ForegroundColor Green
}

# --- main ---
Write-Host "=== Eirin EC2 recover + deploy ($HostIp) ===" -ForegroundColor Cyan

if (-not $DeployOnly) {
  if (-not (Test-TcpPort -HostName $HostIp -Port 22)) {
    Ensure-Ec2Reachable
  } else {
    Write-Host "SSH port already open." -ForegroundColor Green
  }
} else {
  if (-not (Test-TcpPort -HostName $HostIp -Port 22)) {
    throw "DeployOnly: SSH port 22 still closed on $HostIp"
  }
}

$deployArgs = @("-SkipRecover")
if ($Quick) { $deployArgs += "-Quick" }
& (Join-Path $LocalRoot "deploy.ps1") @deployArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Recover + deploy finished." -ForegroundColor Green
