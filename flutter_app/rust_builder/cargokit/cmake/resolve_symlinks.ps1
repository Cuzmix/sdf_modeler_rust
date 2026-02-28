function Resolve-Symlinks {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Position = 0, Mandatory, ValueFromPipeline, ValueFromPipelineByPropertyName)]
        [string] $Path
    )

    [string] $separator = '/'
    [string[]] $parts = $Path.Split($separator)

    [string] $realPath = ''
    foreach ($part in $parts) {
        if ($realPath -and !$realPath.EndsWith($separator)) {
            $realPath += $separator
        }

        $realPath += $part.Replace('\', '/')

        # The slash is important when using Get-Item on Drive letters in pwsh.
        if (-not($realPath.Contains($separator)) -and $realPath.EndsWith(':')) {
            $realPath += '/'
        }

        $item = Get-Item $realPath
        # .LinkTarget is PowerShell 6+; .Target works on PowerShell 5.1
        if ($item.LinkTarget) {
            $realPath = $item.LinkTarget.Replace('\', '/')
        } elseif ($item.Target) {
            $realPath = ($item.Target[0]).Replace('\', '/')
        }
    }
    $realPath
}

$path = Resolve-Symlinks -Path $args[0]
Write-Host $path
