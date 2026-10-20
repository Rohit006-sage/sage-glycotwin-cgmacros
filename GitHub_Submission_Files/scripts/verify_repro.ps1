param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$version = & $Python --version 2>&1
if ($version -notmatch "^Python 3\.11\.") {
    throw "Use Python 3.11.x. Detected: $version"
}

& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Installed package dependencies are inconsistent." }

& $Python scripts/check_reported_metrics.py
if ($LASTEXITCODE -ne 0) { throw "Published result metrics do not match aggregate_metrics.json." }

& $Python scripts/check_docs.py
if ($LASTEXITCODE -ne 0) { throw "Documentation links or dashboard wording are invalid." }

& $Python -m pytest -q --basetemp .pytest-tmp
if ($LASTEXITCODE -ne 0) {
    throw "Test suite failed. The full pipeline contract test requires local CGMacros data under data/raw/cgmacros/."
}

Write-Output "Environment, reported metrics, and full test suite verified."
