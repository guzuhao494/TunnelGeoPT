[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-CommandInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Candidates
    )

    foreach ($candidate in $Candidates) {
        $command = Get-Command -Name $candidate -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return [pscustomobject]@{
                found = $true
                name = $command.Name
                path = $command.Source
            }
        }
    }

    return [pscustomobject]@{
        found = $false
        name = $null
        path = $null
    }
}

function Invoke-VersionCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @("--version")
    )

    try {
        $output = & $FilePath @Arguments 2>$null
        if ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) {
            return (($output | Select-Object -First 1) -join "").Trim()
        }
    }
    catch {
    }

    return $null
}

function Get-PythonInfo {
    $commandInfo = Get-CommandInfo -Candidates @("python", "python3", "py")
    if (-not $commandInfo.found) {
        return [pscustomobject]@{
            found = $false
            launcher = $null
            path = $null
            version = $null
        }
    }

    $args = @("--version")
    if ($commandInfo.name -eq "py") {
        $args = @("-3", "--version")
    }

    return [pscustomobject]@{
        found = $true
        launcher = $commandInfo.name
        path = $commandInfo.path
        version = Invoke-VersionCommand -FilePath $commandInfo.path -Arguments $args
    }
}

function Get-GitInfo {
    $commandInfo = Get-CommandInfo -Candidates @("git")
    if (-not $commandInfo.found) {
        return [pscustomobject]@{
            found = $false
            path = $null
            version = $null
        }
    }

    return [pscustomobject]@{
        found = $true
        path = $commandInfo.path
        version = Invoke-VersionCommand -FilePath $commandInfo.path -Arguments @("--version")
    }
}

function Get-GpuInfo {
    $nvidiaInfo = Get-CommandInfo -Candidates @("nvidia-smi")
    if ($nvidiaInfo.found) {
        try {
            $rows = & $nvidiaInfo.path --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>$null
            $gpus = @()
            foreach ($row in $rows) {
                $parts = $row -split ","
                if ($parts.Count -ge 3) {
                    $gpus += [pscustomobject]@{
                        name = $parts[0].Trim()
                        driver_version = $parts[1].Trim()
                        memory_mb = [int]($parts[2].Trim())
                    }
                }
            }

            return [pscustomobject]@{
                detected = ($gpus.Count -gt 0)
                source = "nvidia-smi"
                gpus = $gpus
            }
        }
        catch {
        }
    }

    try {
        $controllers = Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop |
            Select-Object @{Name = "name"; Expression = { $_.Name } },
                          @{Name = "driver_version"; Expression = { $_.DriverVersion } }

        return [pscustomobject]@{
            detected = ($controllers.Count -gt 0)
            source = "Win32_VideoController"
            gpus = @($controllers)
        }
    }
    catch {
        return [pscustomobject]@{
            detected = $false
            source = "unavailable"
            gpus = @()
        }
    }
}

function Get-WslInfo {
    $commandInfo = Get-CommandInfo -Candidates @("wsl")
    if (-not $commandInfo.found) {
        return [pscustomobject]@{
            visible = $false
            path = $null
            distros = @()
            raw = @()
        }
    }

    try {
        $raw = & $commandInfo.path -l -v 2>$null
        $distros = @()
        $normalizedRaw = @($raw | ForEach-Object { $_ -replace "`0", "" })
        foreach ($line in $normalizedRaw | Select-Object -Skip 1) {
            $trimmed = ($line -replace "^\*", "").Trim()
            if ([string]::IsNullOrWhiteSpace($trimmed)) {
                continue
            }

            $parts = $trimmed -split "\s{2,}"
            if ($parts.Count -ge 3) {
                $distros += [pscustomobject]@{
                    name = $parts[0].Trim()
                    state = $parts[1].Trim()
                    version = $parts[2].Trim()
                }
            }
        }

        return [pscustomobject]@{
            visible = $true
            path = $commandInfo.path
            distros = $distros
            raw = $normalizedRaw
        }
    }
    catch {
        return [pscustomobject]@{
            visible = $true
            path = $commandInfo.path
            distros = @()
            raw = @()
        }
    }
}

$result = [ordered]@{
    generated_at = (Get-Date).ToString("o")
    script = "check_environment.ps1"
    host = [ordered]@{
        os = [ordered]@{
            platform = [System.Environment]::OSVersion.Platform.ToString()
            version = [System.Environment]::OSVersion.VersionString
            is_64bit = [System.Environment]::Is64BitOperatingSystem
        }
        powershell = [ordered]@{
            edition = $PSVersionTable.PSEdition
            version = $PSVersionTable.PSVersion.ToString()
        }
    }
    tools = [ordered]@{
        python = Get-PythonInfo
        git = Get-GitInfo
    }
    gpu = Get-GpuInfo
    wsl = Get-WslInfo
    notes = @(
        "This report intentionally omits environment variables and credential-bearing settings.",
        "A tool being visible on PATH does not prove solver usability or license availability."
    )
}

$parent = Split-Path -Parent $OutputPath
if (-not [string]::IsNullOrWhiteSpace($parent) -and -not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
}

$json = $result | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $OutputPath -Value $json -Encoding utf8
