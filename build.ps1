$ErrorActionPreference = 'Stop'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
& $compiler /nologo /target:winexe /platform:x64 /optimize+ /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.Web.Extensions.dll /reference:System.Net.Http.dll "/out:$PSScriptRoot\오수핵.exe" "$PSScriptRoot\오수핵.cs"
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
