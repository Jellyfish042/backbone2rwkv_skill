param(
    [ValidateSet("zh", "en")]
    [string]$Lang = $(if ($env:BACKBONE2RWKV_LANG) { $env:BACKBONE2RWKV_LANG } else { "zh" }),

    [string]$ProjectRoot = $(Get-Location).Path,

    [string]$Ref = $(if ($env:BACKBONE2RWKV_REF) { $env:BACKBONE2RWKV_REF } else { "main" })
)

$ErrorActionPreference = "Stop"

$Repo = "Jellyfish042/backbone2rwkv_skill"
$SkillName = "backbone2rwkv"
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("backbone2rwkv_skill_" + [System.Guid]::NewGuid().ToString("N"))
$ZipPath = Join-Path $TempRoot "source.zip"

try {
    New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

    $Url = "https://github.com/$Repo/archive/refs/heads/$Ref.zip"
    Write-Host "Downloading $Repo@$Ref..."
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing

    Expand-Archive -LiteralPath $ZipPath -DestinationPath $TempRoot -Force
    $SourceRoot = Get-ChildItem -LiteralPath $TempRoot -Directory |
        Where-Object { $_.Name -like "backbone2rwkv_skill-*" } |
        Select-Object -First 1

    if (-not $SourceRoot) {
        throw "Could not find extracted repository directory."
    }

    $SourceSkill = Join-Path $SourceRoot.FullName "$Lang\$SkillName"
    if (-not (Test-Path -LiteralPath (Join-Path $SourceSkill "SKILL.md"))) {
        throw "Could not find skill at '$SourceSkill'. Check that language '$Lang' exists."
    }

    $SkillsDir = Join-Path $ProjectRoot ".codex\skills"
    $DestSkill = Join-Path $SkillsDir $SkillName

    New-Item -ItemType Directory -Force -Path $SkillsDir | Out-Null
    if (Test-Path -LiteralPath $DestSkill) {
        Remove-Item -LiteralPath $DestSkill -Recurse -Force
    }

    Copy-Item -LiteralPath $SourceSkill -Destination $DestSkill -Recurse -Force

    Write-Host "Installed '$SkillName' ($Lang) to:"
    Write-Host "  $DestSkill"
}
finally {
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
