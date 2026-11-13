# ============================================================
# VideoLingo browser_bridge jobs 缓存清理（方案B：只删缓存，留成品）
# 已实际执行验证：308 目录 / 48.03 GB / 全部成功
# 用法：确认 $dryRun = $false 后执行；执行前须关闭 browser_bridge 服务
# ============================================================

Add-Type -AssemblyName Microsoft.VisualBasic
$ErrorActionPreference = 'Stop'

$jobsRoot = 'D:\My\videolingo-live-dubber\runtime\browser_bridge\jobs'
# 排除最新任务（正在使用/未完成），新任务出现时按需追加
$excludeJobs = @('4c2ff8d769a1')
# 只删这两个子目录。preview 必须保留：含 .confirmed_voice_map.json 音色确认状态
$cacheSubdirs = @('audio','gpt_log')
$logPath = 'D:\My\videolingo-live-dubber\runtime\browser_bridge\_cleanup_log.txt'

# ---------- 前置检查 1：8765 不能被占用 ----------
$conn = Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue
if ($conn) { Write-Host "!!! 中止：browser_bridge 正在运行，请先关闭服务 !!!" -ForegroundColor Red; exit 1 }

# ---------- 前置检查 2：done 任务成品必须齐全 ----------
$indexFile = 'D:\My\videolingo-live-dubber\runtime\browser_bridge\index.json'
if (-not (Test-Path -LiteralPath $indexFile)) { Write-Host "!!! 找不到 index.json，中止 !!!" -ForegroundColor Red; exit 1 }
$idx = Get-Content -LiteralPath $indexFile -Raw | ConvertFrom-Json
$broken = @()
foreach ($j in ($idx.jobs | Where-Object { $_.status -eq "done" })) {
    if ($excludeJobs -contains $j.id) { continue }
    $jd = Join-Path $jobsRoot $j.id
    $srt = Test-Path (Join-Path $jd "dub.srt")
    $ln  = Test-Path (Join-Path $jd "dub_loudnorm.mp3")
    $dub = Test-Path (Join-Path $jd "dub.mp3")
    if (-not ($srt -and ($ln -or $dub))) { $broken += "$($j.id) srt=$srt ln=$ln dub=$dub" }
}
if ($broken.Count -gt 0) {
    Write-Host "!!! 中止：以下任务缺少成品，清理会丢内容 !!!" -ForegroundColor Red
    $broken | ForEach-Object { Write-Host ("    " + $_) }
    exit 1
}

# ---------- 收集待删 ----------
$targets = @()
foreach ($d in (Get-ChildItem -LiteralPath $jobsRoot -Directory -Force)) {
    if ($excludeJobs -contains $d.Name) { continue }
    $wh = Join-Path $d.FullName "work_output"
    if (-not (Test-Path -LiteralPath $wh)) { continue }
    foreach ($sub in $cacheSubdirs) {
        $p = Join-Path $wh $sub
        if (Test-Path -LiteralPath $p) { $targets += (Get-Item -LiteralPath $p) }
    }
}

# ---------- 统计 ----------
$tf = 0; $ts = 0
foreach ($t in $targets) {
    $f = Get-ChildItem -LiteralPath $t.FullName -Recurse -File -Force -ErrorAction SilentlyContinue
    $s = ($f | Measure-Object Length -Sum).Sum; if ($null -eq $s) { $s = 0 }
    $tf += $f.Count; $ts += $s
}
Write-Host "=============== 统计 ==============="
Write-Host ("待删目录 : {0} 个" -f $targets.Count)
Write-Host ("待删文件 : {0} 个" -f $tf)
Write-Host ("待删大小 : {0:N2} GB" -f ($ts/1GB))
Write-Host "保留：dub* 成品 / preview / log / work_output 根散落 / cookies.txt"

$dryRun = $true
if ($dryRun) { Write-Host "`n[空跑] 未做任何修改。改 dryRun 为 $false 才执行。" -ForegroundColor Yellow; exit 0 }

# ---------- 执行：走回收站 ----------
$ok = 0; $fail = @()
foreach ($t in $targets) {
    try {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory(
            $t.FullName,
            [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
            [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)
        $ok++
    } catch { $fail += ("DIR  " + $t.FullName + "  ::  " + $_.Exception.Message) }
}
Write-Host ("完成：成功 $ok / $($targets.Count)，失败 $($fail.Count)")
if ($fail.Count -gt 0) { $fail | Select-Object -First 20 | ForEach-Object { Write-Host ("  " + $_) } }
$fail -join [Environment]::NewLine | Set-Content -LiteralPath $logPath -Encoding UTF8
