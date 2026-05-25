param(
    [ValidateSet("zh", "en")]
    [string]$Lang = $(if ($env:BACKBONE2RWKV_LANG) { $env:BACKBONE2RWKV_LANG } else { "zh" }),

    [ValidateSet("backbone2rwkv", "optimize-rwkv7")]
    [string]$Skill = $(if ($env:BACKBONE2RWKV_SKILL) { $env:BACKBONE2RWKV_SKILL } else { "backbone2rwkv" }),

    [string]$ProjectRoot = $(Get-Location).Path,

    [string]$Ref = $(if ($env:BACKBONE2RWKV_REF) { $env:BACKBONE2RWKV_REF } else { "main" })
)

$ErrorActionPreference = "Stop"

$Repo = "Jellyfish042/backbone2rwkv_skill"
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("backbone2rwkv_skill_" + [System.Guid]::NewGuid().ToString("N"))
$ZipPath = Join-Path $TempRoot "source.zip"

switch ($Skill) {
    "backbone2rwkv" {
        $SourceRelativePath = "backbone2rwkv_$Lang\backbone2rwkv"
        $DestName = "backbone2rwkv"
    }
    "optimize-rwkv7" {
        $SourceRelativePath = "optimize_rwkv7_$Lang\optimize-rwkv7"
        $DestName = "optimize-rwkv7"
    }
}

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

    $SourceSkill = Join-Path $SourceRoot.FullName $SourceRelativePath
    if (-not (Test-Path -LiteralPath (Join-Path $SourceSkill "SKILL.md"))) {
        throw "Could not find skill at '$SourceSkill'. Check that language '$Lang' and skill '$Skill' exist."
    }

    $SkillsDir = Join-Path $ProjectRoot ".codex\skills"
    $DestSkill = Join-Path $SkillsDir $DestName

    New-Item -ItemType Directory -Force -Path $SkillsDir | Out-Null
    if (Test-Path -LiteralPath $DestSkill) {
        Remove-Item -LiteralPath $DestSkill -Recurse -Force
    }

    Copy-Item -LiteralPath $SourceSkill -Destination $DestSkill -Recurse -Force

    Write-Host "Installed '$DestName' ($Lang) to:"
    Write-Host "  $DestSkill"
}
finally {
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
