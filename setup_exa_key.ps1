# Run locally: .\setup_exa_key.ps1
# Keep the key out of terminal history and chat.
$secureExaKey = Read-Host 'Exa API key' -AsSecureString
$exaPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureExaKey)
try {
    $newExaKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($exaPointer)
    if ($newExaKey -notmatch '^[A-Za-z0-9_-]+$') {
        throw 'The key must contain only letters, digits, hyphens, or underscores.'
    }
    $exaEnvPath = Join-Path $PSScriptRoot '.env'
    $exaEnvText = if (Test-Path -LiteralPath $exaEnvPath) { [IO.File]::ReadAllText($exaEnvPath) } else { '' }
    if ($exaEnvText -match '(?m)^EXA_API_KEY=.*$') {
        $exaEnvText = [regex]::Replace($exaEnvText, '(?m)^EXA_API_KEY=.*$', "EXA_API_KEY=$newExaKey")
    } else {
        $exaEnvText += "`nEXA_API_KEY=$newExaKey`n"
    }
    [IO.File]::WriteAllText($exaEnvPath, $exaEnvText, [Text.UTF8Encoding]::new($false))
    Write-Output 'Exa key saved in the local .env file.'
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($exaPointer)
    $newExaKey = $null
    $exaEnvText = $null
}
