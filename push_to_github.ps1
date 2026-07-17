Param(
    [string]$Message = ""
)

if (-not $Message) {
    $Message = "update: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

# Check for changes
$status = git status --porcelain
if (-not $status) {
    Write-Output "No changes to commit."
    exit 0
}

git add -A
git commit -m $Message

# Push to origin master (adjust branch if you use a different one)
git push origin master
