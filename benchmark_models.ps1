# 模型基准测试脚本 - 每个模型运行100次

# 配置参数
$executables = @(
    @{
        Name = "ONNX_Runtime"
        Path = "D:\ertek_codebase\onnx_demo\build\bin\Release\batch_inference.exe"
    },
    @{
        Name = "OpenVINO"
        Path = "D:\ertek_codebase\onnx_demo\build\bin\Release\batch_inference_openvino.exe"
    }
)

$imagesDir = "D:\ertek_data\scratch_data\images"
$testList = "D:\ertek_codebase\onnx_demo\test.txt"
$outputDir = "C:\Users\licha\Desktop\benchmark_output"
$timingDir = "C:\Users\licha\Desktop\benchmark_output"
$enable_save = 0  # 设置为1以启用结果保存，0为禁用

# 模型列表
$models = @(
    @{
        Name = "LiteU_Net"
        Path = "D:\ertek_codebase\scratch_detection\LiteU_Net_checkpoints_20251224_125726\best.onnx"
    },
    @{
        Name = "MicroU_Net"
        Path = "D:\ertek_codebase\scratch_detection\MicroU_Net_checkpoints_20251225_094443\best_fixed.onnx"
    },
    @{
        Name = "NanoU_Net"
        Path = "D:\ertek_codebase\scratch_detection\NanoU_Net_checkpoints_20251225_134817\best.onnx"
    },
    @{
        Name = "TinyU_Net"
        Path = "D:\ertek_codebase\scratch_detection\TinyU_Net_checkpoints_20251225_150721\best.onnx"
    },
    @{
        Name = "StandardU_Net"
        Path = "D:\ertek_codebase\onnx_demo\models\model.onnx"
    }
)

# 运行次数
$iterations = 100

# 创建输出目录
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "模型基准测试" -ForegroundColor Cyan
Write-Host "每个模型运行次数: $iterations" -ForegroundColor Cyan
Write-Host "测试引擎: ONNX Runtime & OpenVINO" -ForegroundColor Cyan
Write-Host "输出目录: $outputDir" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 遍历每个推理引擎
foreach ($exe in $executables) {
    $engineName = $exe.Name
    $exePath = $exe.Path
    
    Write-Host "`n======== 测试引擎: $engineName ========" -ForegroundColor Magenta
    
    # 检查可执行文件是否存在
    if (-not (Test-Path $exePath)) {
        Write-Host "错误: 可执行文件不存在，跳过: $exePath" -ForegroundColor Red
        continue
    }

# 遍历每个模型
foreach ($model in $models) {
    $modelName = $model.Name
    $modelPath = $model.Path
    $timingFile = Join-Path $timingDir "$($engineName)_$($modelName)_timing.txt"
    $errorFile = Join-Path $timingDir "$($engineName)_$($modelName)_errors.txt"
    
    Write-Host "开始测试模型: $modelName (引擎: $engineName)" -ForegroundColor Yellow
    Write-Host "模型路径: $modelPath" -ForegroundColor Gray
    
    # 检查模型是否存在
    if (-not (Test-Path $modelPath)) {
        Write-Host "错误: 模型文件不存在，跳过" -ForegroundColor Red
        continue
    }
    
    # 初始化累计统计
    $totalInferenceTimes = @()
    $successCount = 0
    $failCount = 0
    $errors = @()
    
    # 清空错误日志文件
    if (Test-Path $errorFile) {
        Remove-Item $errorFile -Force
    }
    
    # 运行100次
    for ($i = 1; $i -le $iterations; $i++) {
        Write-Host "`r[$engineName - $modelName] 运行: $i/$iterations " -NoNewline -ForegroundColor Cyan
        
        try {
            # 执行推理，捕获错误输出
            $startTime = Get-Date
            $errorOutput = & $exePath `
                --model $modelPath `
                --images $imagesDir `
                --list $testList `
                --output $outputDir `
                --timing "$timingFile.temp" `
                --save $enable_save 2>&1 | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] -or $_.ToString() -match "ERROR|Error|error" }
            $exitCode = $LASTEXITCODE
            $endTime = Get-Date
            $execTime = ($endTime - $startTime).TotalSeconds
            
            if ($exitCode -eq 0) {
                $successCount++
                
                # 读取临时timing文件获取总推理时间
                if (Test-Path "$timingFile.temp") {
                    $timingContent = Get-Content "$timingFile.temp"
                    $totalLine = $timingContent | Where-Object { $_ -match "Total inference time: ([\d.]+) ms" }
                    if ($totalLine -match "([\d.]+) ms") {
                        $totalInferenceTimes += [double]$matches[1]
                    }
                }
            } else {
                $failCount++
                $errorMsg = "运行 $i 失败 (退出码: $exitCode)"
                $errors += $errorMsg
                
                # 记录详细错误信息
                Add-Content -Path $errorFile -Value "========== 运行 $i =========="
                Add-Content -Path $errorFile -Value "退出码: $exitCode"
                Add-Content -Path $errorFile -Value "执行时间: $([Math]::Round($execTime, 2))s"
                Add-Content -Path $errorFile -Value ""
                
                if ($errorOutput) {
                    Add-Content -Path $errorFile -Value "错误输出:"
                    Add-Content -Path $errorFile -Value "----------------------------------------"
                    $errorOutput | ForEach-Object { Add-Content -Path $errorFile -Value $_.ToString() }
                } else {
                    Add-Content -Path $errorFile -Value "未捕获到具体错误信息"
                }
                Add-Content -Path $errorFile -Value ""
                
                Write-Host "`r[$engineName - $modelName] 运行: $i/$iterations - 失败! (退出码: $exitCode)     " -ForegroundColor Red
            }
        } catch {
            $failCount++
            $errorMsg = "运行 $i 异常: $_"
            $errors += $errorMsg
            
            # 记录异常信息
            Add-Content -Path $errorFile -Value "========== 运行 $i =========="
            Add-Content -Path $errorFile -Value "异常类型: $($_.Exception.GetType().FullName)"
            Add-Content -Path $errorFile -Value "异常消息: $($_.Exception.Message)"
            Add-Content -Path $errorFile -Value "异常类型: $($_.Exception.GetType().FullName)"
            Add-Content -Path $errorFile -Value "异常消息: $($_.Exception.Message)"
            if ($_.Exception.StackTrace) {
                Add-Content -Path $errorFile -Value ""
                Add-Content -Path $errorFile -Value "堆栈跟踪:"
                Add-Content -Path $errorFile -Value $_.Exception.StackTrace
            }
            Add-Content -Path $errorFile -Value ""
            Write-Host "`r[$engineName - $modelName] 运行: $i/$iterations - 异常!     " -ForegroundColor Red
        }
    }
    
    Write-Host "`r[$engineName - $modelName] 完成! 成功: $successCount, 失败: $failCount                    " -ForegroundColor Green
    
    # 计算统计结果
    if ($totalInferenceTimes.Count -gt 0) {
        $avgTime = ($totalInferenceTimes | Measure-Object -Average).Average
        $minTime = ($totalInferenceTimes | Measure-Object -Minimum).Minimum
        $maxTime = ($totalInferenceTimes | Measure-Object -Maximum).Maximum
        $stdDev = [Math]::Sqrt((($totalInferenceTimes | ForEach-Object { [Math]::Pow($_ - $avgTime, 2) }) | Measure-Object -Average).Average)
        
        # 写入汇总报告
        $report = @"
========================================
模型基准测试报告
========================================
推理引擎: $engineName
模型名称: $modelName
模型路径: $modelPath
运行次数: $iterations
成功次数: $successCount
失败次数: $failCount

总推理时间统计 (ms):
----------------------------------------
平均值: $([Math]::Round($avgTime, 2))
最小值: $([Math]::Round($minTime, 2))
最大值: $([Math]::Round($maxTime, 2))
标准差: $([Math]::Round($stdDev, 2))

详细数据:
----------------------------------------
"@
        
        Set-Content -Path $timingFile -Value $report
        
        # 添加每次运行的详细数据
        for ($i = 0; $i -lt $totalInferenceTimes.Count; $i++) {
            Add-Content -Path $timingFile -Value "运行 $($i+1): $([Math]::Round($totalInferenceTimes[$i], 2)) ms"
        }
        
        # 添加错误信息
        if ($errors.Count -gt 0) {
            Add-Content -Path $timingFile -Value "`n========================================"
            Add-Content -Path $timingFile -Value "错误摘要:"
            Add-Content -Path $timingFile -Value "========================================"
            foreach ($error in $errors) {
                Add-Content -Path $timingFile -Value $error
            }
            Add-Content -Path $timingFile -Value "`n详细错误信息保存在: $errorFile"
        }
        
        Write-Host "平均推理时间: $([Math]::Round($avgTime, 2)) ms ± $([Math]::Round($stdDev, 2)) ms" -ForegroundColor Green
        Write-Host "Timing报告已保存: $timingFile" -ForegroundColor Gray
        if ($errors.Count -gt 0) {
            Write-Host "错误日志已保存: $errorFile" -ForegroundColor Yellow
        }
        Write-Host ""
    } else {
        Write-Host "警告: 没有有效的推理时间数据" -ForegroundColor Red
    }
    
    # 清理临时文件
    if (Test-Path "$timingFile.temp") {
        Remove-Item "$timingFile.temp" -Force
    }
}
}

# 生成总结报告
$summaryFile = Join-Path $timingDir "benchmark_summary.txt"
$summary = @"
========================================
模型基准测试总结
========================================
测试时间: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
每个模型运行次数: $iterations
输出目录: $outputDir

测试结果 (总推理时间平均值, ms):
"@

Set-Content -Path $summaryFile -Value $summary

foreach ($exe in $executables) {
    $engineName = $exe.Name
    Add-Content -Path $summaryFile -Value "`n[$engineName]"
    Add-Content -Path $summaryFile -Value "----------------------------------------"
    
    foreach ($model in $models) {
        $modelName = $model.Name
        $timingFile = Join-Path $timingDir "$($engineName)_$($modelName)_timing.txt"
        
        if (Test-Path $timingFile) {
            $content = Get-Content $timingFile
            $avgLine = $content | Where-Object { $_ -match "平均值: ([\d.]+)" }
            $successLine = $content | Where-Object { $_ -match "成功次数: (\d+)" }
            $failLine = $content | Where-Object { $_ -match "失败次数: (\d+)" }
            
            if ($avgLine -and $successLine) {
                $avgMatch = $avgLine -match "([\d.]+)"
                $successMatch = $successLine -match "(\d+)"
                $failMatch = $failLine -match "(\d+)"
                Add-Content -Path $summaryFile -Value "  $modelName : $($matches[1]) ms (成功: $successMatch 次, 失败: $failMatch 次)"
            }
        }
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "所有测试完成!" -ForegroundColor Green
Write-Host "总结报告: $summaryFile" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
