# Deploy on AWS - bind 0.0.0.0 so app is reachable at http://3.151.143.63:8000
# Run on the EC2 instance: .\deploy_aws.ps1   or   pwsh -File deploy_aws.ps1

$env:DEPLOY_AWS = "1"
$env:PORT = "8000"
& "$PSScriptRoot\deploy.ps1"
