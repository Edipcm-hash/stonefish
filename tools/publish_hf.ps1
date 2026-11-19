# Upload the StonefishAI weights to Hugging Face over git + LFS.
#
# Why a script: the checkpoint is 848 MB, so it needs git-lfs, and the Hub repo has
# to exist before the push. Doing it by hand is four commands that are easy to get
# wrong, and the failure mode (a 130-byte pointer file instead of the model) looks
# like success until someone tries to download it.
#
# The token is never stored: `git push` prompts for it and git keeps it in the
# credential manager, not in this script.
#
#   .\tools\publish_hf.ps1                      # defaults to Edipcm-hash/stonefish-222m
#   .\tools\publish_hf.ps1 -Repo someone/other

param(
    [string]$Repo = "Edipcm-hash/stonefish-222m",
    [string]$User = "Edipcm-hash",
    [string]$State = "stonefish_en.final.state",
    [string]$Vocab = "data/en/tiny.bpe",
    [string]$Card = "docs/huggingface-README.md"
)

$ErrorActionPreference = "Stop"

function Say($msg) { Write-Host "  $msg" }

Write-Host "`n=== StonefishAI -> Hugging Face`n"

# ---- prerequisites -----------------------------------------------------------
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git not found" }
if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) {
    throw "git-lfs not found. Install it, then re-run: https://git-lfs.com"
}
foreach ($f in @($State, $Vocab, $Card)) {
    if (-not (Test-Path $f)) { throw "missing file: $f (run from the repository root)" }
}
Say "prerequisites ok"

# ---- tell the user what will happen -------------------------------------------
$stateMb = [math]::Round((Get-Item $State).Length / 1MB, 0)
$vocabKb = [math]::Round((Get-Item $Vocab).Length / 1KB, 0)
Write-Host ""
Say "repo  : $Repo"
Say "model : $State ($stateMb MB)"
Say "vocab : $Vocab ($vocabKb KB) - the checkpoint alone cannot decode"
Write-Host ""

if (-not (Test-Path $Repo.Split('/')[-1])) {
    Say "cloning https://huggingface.co/$Repo ..."
    git clone "https://huggingface.co/$Repo" $Repo.Split('/')[-1]
}
$dir = Join-Path (Get-Location) $Repo.Split('/')[-1]
Push-Location $dir
try {
    git lfs install --local | Out-Null
    git lfs track "*.state" | Out-Null
    Say "git-lfs is tracking *.state"

    Copy-Item (Join-Path (Split-Path -Parent (Get-Location)) $State) . -Force
    Copy-Item (Join-Path (Split-Path -Parent (Get-Location)) $Vocab) . -Force
    Copy-Item (Join-Path (Split-Path -Parent (Get-Location)) $Card) "README.md" -Force
    Say "files copied (state, tiny.bpe, README.md)"

    git add .gitattributes README.md tiny.bpe
    git add $State
    git commit -m "StonefishAI v2 - 222M parameters, 2.05B tokens, zero dependencies" 2>&1 | Out-Null
    Say "committed"

    Write-Host ""
    Write-Host "  When git asks:" -ForegroundColor Yellow
    Write-Host "    Username : $User" -ForegroundColor Yellow
    Write-Host "    Password : your hf_... token (it is not echoed, and it is not saved here)" -ForegroundColor Yellow
    Write-Host ""
    git push
    if ($LASTEXITCODE -ne 0) { throw "push failed - check the username ($User) and that the token has Write access" }

    Write-Host ""
    Say "pushed: https://huggingface.co/$Repo"
    Say "verify: the file page must show a real size, not a 130-byte pointer"
}
finally {
    Pop-Location
}
