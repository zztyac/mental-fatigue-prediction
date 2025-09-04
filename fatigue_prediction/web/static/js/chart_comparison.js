// 模型对比图表生成函数
function initModelComparisonCharts(modelData) {
    // 获取模型名称
    const modelNames = Object.keys(modelData.results);
    
    // 提取各模型的指标数据
    const trainLosses = modelNames.map(model => ({
        name: model,
        value: modelData.results[model].final_train_loss
    }));
    
    const valLosses = modelNames.map(model => ({
        name: model,
        value: modelData.results[model].final_test_loss
    }));
    
    const r2Scores = modelNames.map(model => ({
        name: model,
        value: modelData.results[model].final_r2
    }));
    
    const trainingTimes = modelNames.map(model => ({
        name: model,
        value: modelData.results[model].training_time
    }));
    
    // 颜色配置
    const colors = {
        'CNN': '#e74c3c',
        'LSTM': '#3498db',
        'Transformer': '#2ecc71'
    };
    
    const defaultColors = ['#e74c3c', '#3498db', '#2ecc71'];
    
    // 创建训练损失和验证损失对比图表
    createLossComparisonChart('loss-comparison-chart', trainLosses, valLosses, colors);
    
    // 创建R²分数对比图表
    createR2ComparisonChart('r2-comparison-chart', r2Scores, colors);
    
    // 创建参数数量对比图表
    createParamsComparisonChart('params-comparison-chart', modelData.params, colors);
    
    // 创建训练时间对比图表
    createTrainingTimeChart('training-time-chart', trainingTimes, colors);
    
    // 创建综合性能雷达图
    createPerformanceRadarChart('performance-radar-chart', modelData, colors);
    
    // 使用详细数据创建训练过程曲线图（如果有详细数据）
    if (modelData.detailed_results && modelData.epochs) {
        // 创建训练和验证损失曲线图
        createLossLinesChart('loss-lines-chart', modelData.detailed_results, modelData.epochs, colors);
        
        // 创建R²分数变化曲线图
        createR2LinesChart('r2-lines-chart', modelData.detailed_results, modelData.epochs, colors);
    }
}

// 损失对比图表
function createLossComparisonChart(containerId, trainLosses, valLosses, colors) {
    const chartContainer = document.getElementById(containerId);
    if (!chartContainer) return;
    
    // 初始化图表实例
    const chart = echarts.init(chartContainer);
    
    // 提取数据
    const modelNames = trainLosses.map(item => item.name);
    const trainLossValues = trainLosses.map(item => item.value);
    const valLossValues = valLosses.map(item => item.value);
    
    // 为每个系列设置颜色
    const trainLossColors = modelNames.map(name => colors[name] || defaultColors[0]);
    const valLossColors = modelNames.map(name => {
        const baseColor = colors[name] || defaultColors[0];
        return echarts.color.lighten(baseColor, 0.3); // 验证损失使用浅色
    });
    
    // 图表配置
    const option = {
        title: {
            text: '训练损失与验证损失对比',
            left: 'center',
            textStyle: {
                color: '#ffffff'
            }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: function(params) {
                const trainLoss = params[0].value.toFixed(4);
                const valLoss = params[1].value.toFixed(4);
                return `<b>${params[0].name}</b><br/>
                        训练损失: ${trainLoss}<br/>
                        验证损失: ${valLoss}`;
            }
        },
        legend: {
            data: ['训练损失', '验证损失'],
            top: 30,
            textStyle: {
                color: '#ffffff'
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: modelNames,
            axisLine: {
                lineStyle: {
                    color: '#ffffff'
                }
            },
            axisLabel: {
                rotate: 0,
                color: '#ffffff'
            }
        },
        yAxis: {
            type: 'value',
            name: '损失值',
            nameTextStyle: {
                color: '#ffffff'
            },
            axisLine: {
                lineStyle: {
                    color: '#ffffff'
                }
            },
            axisLabel: {
                color: '#ffffff',
                formatter: function(value) {
                    return value.toFixed(2);
                }
            },
            splitLine: {
                lineStyle: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            }
        },
        series: [
            {
                name: '训练损失',
                type: 'bar',
                data: trainLossValues,
                itemStyle: {
                    color: function(params) {
                        return trainLossColors[params.dataIndex];
                    }
                },
                label: {
                    show: true,
                    position: 'top',
                    color: '#ffffff',
                    formatter: function(params) {
                        return params.value.toFixed(4);
                    }
                },
                barWidth: '30%',
                barGap: '0%'
            },
            {
                name: '验证损失',
                type: 'bar',
                data: valLossValues,
                itemStyle: {
                    color: function(params) {
                        return valLossColors[params.dataIndex];
                    }
                },
                label: {
                    show: true,
                    position: 'top',
                    color: '#ffffff',
                    formatter: function(params) {
                        return params.value.toFixed(4);
                    }
                },
                barWidth: '30%',
                barGap: '0%'
            }
        ],
        backgroundColor: 'transparent'
    };
    
    // 应用配置
    chart.setOption(option);
    
    // 响应式调整
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// R²分数对比图表
function createR2ComparisonChart(containerId, r2Scores, colors) {
    const chartContainer = document.getElementById(containerId);
    if (!chartContainer) return;
    
    // 初始化图表实例
    const chart = echarts.init(chartContainer);
    
    // 提取数据
    const modelNames = r2Scores.map(item => item.name);
    const r2Values = r2Scores.map(item => item.value);
    
    // 生成渐变色
    const gradientColors = modelNames.map(name => {
        const baseColor = colors[name] || defaultColors[0];
        return new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: echarts.color.lighten(baseColor, 0.2) },
            { offset: 1, color: baseColor }
        ]);
    });
    
    // 图表配置
    const option = {
        title: {
            text: 'R²分数对比（越接近1越好）',
            left: 'center',
            textStyle: {
                color: '#ffffff'
            }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: function(params) {
                return `<b>${params[0].name}</b><br/>
                        R²: ${params[0].value.toFixed(4)}`;
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: modelNames,
            axisLine: {
                lineStyle: {
                    color: '#ffffff'
                }
            },
            axisLabel: {
                color: '#ffffff'
            }
        },
        yAxis: {
            type: 'value',
            name: 'R²值',
            nameTextStyle: {
                color: '#ffffff'
            },
            min: function(value) {
                // 确保y轴下限不高于最小的R²值
                return Math.min(-0.5, value.min);
            },
            max: function(value) {
                // 尝试把1.0也包括在范围内，便于参考
                return Math.max(1.0, value.max);
            },
            axisLine: {
                lineStyle: {
                    color: '#ffffff'
                }
            },
            axisLabel: {
                color: '#ffffff'
            },
            splitLine: {
                lineStyle: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            }
        },
        series: [
            {
                type: 'bar',
                data: r2Values.map((value, index) => ({
                    value: value,
                    itemStyle: {
                        color: gradientColors[index]
                    }
                })),
                label: {
                    show: true,
                    position: 'top',
                    color: '#ffffff',
                    formatter: function(params) {
                        return params.value.toFixed(4);
                    }
                },
                barWidth: '50%'
            },
            {
                // 添加一条R²=1的参考线
                type: 'line',
                markLine: {
                    silent: true,
                    lineStyle: {
                        color: '#ffffff',
                        type: 'dashed'
                    },
                    data: [
                        {
                            yAxis: 1,
                            label: {
                                show: true,
                                position: 'end',
                                formatter: 'R²=1（完美拟合）',
                                color: '#ffffff'
                            }
                        }
                    ]
                }
            }
        ],
        backgroundColor: 'transparent'
    };
    
    // 应用配置
    chart.setOption(option);
    
    // 响应式调整
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 参数数量对比图表
function createParamsComparisonChart(containerId, paramsData, colors) {
    const chartContainer = document.getElementById(containerId);
    if (!chartContainer) return;
    
    // 初始化图表实例
    const chart = echarts.init(chartContainer);
    
    // 提取数据
    const modelNames = Object.keys(paramsData);
    // 移除逗号，转换为数字
    const paramsValues = modelNames.map(name => {
        let rawValue = paramsData[name];
        if (typeof rawValue === 'string') {
            return parseInt(rawValue.replace(/,/g, ''));
        }
        return rawValue;
    });
    
    // 生成渐变色
    const gradientColors = modelNames.map(name => {
        const baseColor = colors[name] || defaultColors[0];
        return new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: echarts.color.lighten(baseColor, 0.3) },
            { offset: 1, color: baseColor }
        ]);
    });
    
    // 图表配置
    const option = {
        title: {
            text: '模型参数数量对比',
            left: 'center',
            textStyle: {
                color: '#ffffff'
            }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: function(params) {
                // 格式化数字，添加千位分隔符
                const formattedValue = params[0].value.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
                return `<b>${params[0].name}</b><br/>
                        参数数量: ${formattedValue}`;
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: modelNames,
            axisLine: {
                lineStyle: {
                    color: '#ffffff'
                }
            },
            axisLabel: {
                color: '#ffffff'
            }
        },
        yAxis: {
            type: 'value',
            name: '参数数量',
            nameTextStyle: {
                color: '#ffffff'
            },
            axisLine: {
                lineStyle: {
                    color: '#ffffff'
                }
            },
            axisLabel: {
                color: '#ffffff',
                formatter: function(value) {
                    // 格式化为K/M
                    if (value >= 1000000) {
                        return (value / 1000000).toFixed(1) + 'M';
                    } else if (value >= 1000) {
                        return (value / 1000).toFixed(0) + 'K';
                    }
                    return value;
                }
            },
            splitLine: {
                lineStyle: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            }
        },
        series: [
            {
                type: 'bar',
                data: paramsValues.map((value, index) => ({
                    value: value,
                    itemStyle: {
                        color: gradientColors[index]
                    }
                })),
                label: {
                    show: true,
                    position: 'top',
                    color: '#ffffff',
                    formatter: function(params) {
                        // 格式化为K/M
                        if (params.value >= 1000000) {
                            return (params.value / 1000000).toFixed(1) + 'M';
                        } else if (params.value >= 1000) {
                            return (params.value / 1000).toFixed(0) + 'K';
                        }
                        return params.value;
                    }
                },
                barWidth: '50%'
            }
        ],
        backgroundColor: 'transparent'
    };
    
    // 应用配置
    chart.setOption(option);
    
    // 响应式调整
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 训练时间对比图表
function createTrainingTimeChart(containerId, trainingTimes, colors) {
    const chartContainer = document.getElementById(containerId);
    if (!chartContainer) return;
    
    // 初始化图表实例
    const chart = echarts.init(chartContainer);
    
    // 提取数据
    const modelNames = trainingTimes.map(item => item.name);
    const timeValues = trainingTimes.map(item => item.value);
    
    // 生成环形图数据
    const pieData = timeValues.map((value, index) => ({
        name: modelNames[index],
        value: value,
        itemStyle: {
            color: colors[modelNames[index]] || defaultColors[index % defaultColors.length]
        }
    }));
    
    // 图表配置
    const option = {
        title: {
            text: '训练时间对比',
            left: 'center',
            textStyle: {
                color: '#ffffff'
            }
        },
        tooltip: {
            trigger: 'item',
            formatter: function(params) {
                // 转换为分钟和秒
                let time = params.value;
                let minutes = Math.floor(time / 60);
                let seconds = (time % 60).toFixed(1);
                let timeStr = "";
                
                if (minutes > 0) {
                    timeStr = `${minutes}分${seconds}秒`;
                } else {
                    timeStr = `${seconds}秒`;
                }
                
                return `<b>${params.name}</b><br/>
                        训练时间: ${timeStr}<br/>
                        占比: ${params.percent}%`;
            }
        },
        legend: {
            orient: 'vertical',
            right: 10,
            top: 'center',
            textStyle: {
                color: '#ffffff'
            }
        },
        series: [
            {
                name: '训练时间',
                type: 'pie',
                radius: ['40%', '70%'],
                avoidLabelOverlap: false,
                label: {
                    show: true,
                    color: '#ffffff',
                    formatter: function(params) {
                        // 转换为分钟和秒的简化版
                        let time = params.value;
                        let minutes = Math.floor(time / 60);
                        let seconds = (time % 60).toFixed(0);
                        let timeStr = "";
                        
                        if (minutes > 0) {
                            timeStr = `${minutes}m${seconds}s`;
                        } else {
                            timeStr = `${seconds}s`;
                        }
                        
                        return `${params.name}\n${timeStr}`;
                    }
                },
                emphasis: {
                    label: {
                        show: true,
                        fontSize: '16',
                        fontWeight: 'bold'
                    }
                },
                data: pieData
            }
        ],
        backgroundColor: 'transparent'
    };
    
    // 应用配置
    chart.setOption(option);
    
    // 响应式调整
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 综合性能雷达图
function createPerformanceRadarChart(containerId, modelData, colors) {
    const chartContainer = document.getElementById(containerId);
    if (!chartContainer) return;
    
    // 初始化图表实例
    const chart = echarts.init(chartContainer);
    
    // 提取数据
    const modelNames = Object.keys(modelData.results);
    
    // 为雷达图准备数据
    // 注意：对于损失值，我们需要取反转换，使得值越小反而得分越高
    const radarData = modelNames.map(name => {
        const model = modelData.results[name];
        
        // 标准化处理：将不同量级的指标转换为0-100的分数
        // 训练损失和验证损失需要转换为"越低越好"的分数
        // R²分数需要标准化到0-100（考虑到可能有负值）
        return {
            name: name,
            value: [
                // 训练损失 - 反转并标准化，越低越好
                calculateInverseScore(model.final_train_loss, modelNames.map(m => modelData.results[m].final_train_loss)),
                // 验证损失 - 反转并标准化，越低越好
                calculateInverseScore(model.final_test_loss, modelNames.map(m => modelData.results[m].final_test_loss)),
                // R²分数 - 标准化到0-100，越高越好
                calculateScore(model.final_r2, modelNames.map(m => modelData.results[m].final_r2), -1, 1),
                // 训练时间 - 反转并标准化，越短越好
                calculateInverseScore(model.training_time, modelNames.map(m => modelData.results[m].training_time))
            ],
            lineStyle: {
                color: colors[name]
            },
            areaStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: echarts.color.modifyAlpha(colors[name], 0.6) },
                    { offset: 1, color: echarts.color.modifyAlpha(colors[name], 0.1) }
                ])
            },
            itemStyle: {
                color: colors[name]
            }
        };
    });
    
    // 图表配置
    const option = {
        title: {
            text: '模型综合性能雷达图',
            left: 'center',
            textStyle: {
                color: '#ffffff'
            }
        },
        tooltip: {
            trigger: 'item'
        },
        legend: {
            data: modelNames,
            bottom: 0,
            textStyle: {
                color: '#ffffff'
            }
        },
        radar: {
            indicator: [
                { name: '训练损失\n(越低越好)', max: 100 },
                { name: '验证损失\n(越低越好)', max: 100 },
                { name: 'R²分数\n(越高越好)', max: 100 },
                { name: '训练时间\n(越短越好)', max: 100 }
            ],
            splitArea: {
                areaStyle: {
                    color: ['rgba(255, 255, 255, 0.05)', 'rgba(255, 255, 255, 0.1)']
                }
            },
            axisLine: {
                lineStyle: {
                    color: 'rgba(255, 255, 255, 0.2)'
                }
            },
            splitLine: {
                lineStyle: {
                    color: 'rgba(255, 255, 255, 0.2)'
                }
            },
            name: {
                textStyle: {
                    color: '#ffffff'
                }
            }
        },
        series: [
            {
                type: 'radar',
                data: radarData
            }
        ],
        backgroundColor: 'transparent'
    };
    
    // 应用配置
    chart.setOption(option);
    
    // 响应式调整
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 辅助函数：计算反转的标准化分数（越低越好的指标）
function calculateInverseScore(value, allValues) {
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    
    // 处理特殊情况
    if (min === max) return 50; // 所有值相等
    if (value === min) return 100; // 最低值
    if (value === max) return 0; // 最高值
    
    // 标准化公式，反转使其越低越好
    return 100 - (((value - min) / (max - min)) * 100);
}

// 辅助函数：计算标准化分数（越高越好的指标）
function calculateScore(value, allValues, absolute_min = null, absolute_max = null) {
    let min = Math.min(...allValues);
    let max = Math.max(...allValues);
    
    // 如果提供了绝对范围，则使用绝对范围
    if (absolute_min !== null) min = Math.min(min, absolute_min);
    if (absolute_max !== null) max = Math.max(max, absolute_max);
    
    // 处理特殊情况
    if (min === max) return 50; // 所有值相等
    if (value === max) return 100; // 最高值
    if (value === min) return 0; // 最低值
    
    // 标准化公式
    return ((value - min) / (max - min)) * 100;
}

// 新增的训练和验证损失曲线图
function createLossLinesChart(containerId, detailedResults, epochs, colors) {
    const chartContainer = document.getElementById(containerId);
    if (!chartContainer) return;
    
    // 初始化图表实例
    const chart = echarts.init(chartContainer);
    
    // 提取数据
    const modelNames = Object.keys(detailedResults);
    const xAxisData = Array.from({length: epochs}, (_, i) => i + 1);
    
    // 准备系列数据
    const series = [];
    
    modelNames.forEach(model => {
        // 训练损失
        series.push({
            name: `${model} 训练损失`,
            type: 'line',
            data: detailedResults[model].train_losses,
            symbol: 'circle',
            symbolSize: 5,
            lineStyle: {
                width: 2,
                color: colors[model]
            },
            itemStyle: {
                color: colors[model]
            }
        });
        
        // 验证损失
        series.push({
            name: `${model} 验证损失`,
            type: 'line',
            data: detailedResults[model].test_losses,
            symbol: 'circle',
            symbolSize: 5,
            lineStyle: {
                width: 2,
                type: 'dashed',
                color: echarts.color.lighten(colors[model], 0.3)
            },
            itemStyle: {
                color: echarts.color.lighten(colors[model], 0.3)
            }
        });
    });
    
    // 图表配置
    const option = {
        title: {
            text: '训练过程中的损失变化',
            left: 'center',
            textStyle: {
                color: '#ffffff'
            }
        },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                let result = `<div>第 ${params[0].axisValue} 轮</div>`;
                params.forEach(param => {
                    result += `<div style="margin: 5px 0;">
                        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background-color:${param.color};"></span>
                        <span style="margin-left:5px">${param.seriesName}: ${param.value.toFixed(6)}</span>
                    </div>`;
                });
                return result;
            }
        },
        legend: {
            data: series.map(s => s.name),
            textStyle: {
                color: '#ffffff'
            },
            top: 30
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: xAxisData,
            name: 'Epoch',
            nameLocation: 'middle',
            nameGap: 30,
            nameTextStyle: {
                color: '#ffffff'
            },
            axisLine: {
                lineStyle: {
                    color: '#ffffff'
                }
            },
            axisLabel: {
                color: '#ffffff'
            }
        },
        yAxis: {
            type: 'value',
            name: '损失',
            nameTextStyle: {
                color: '#ffffff'
            },
            axisLine: {
                lineStyle: {
                    color: '#ffffff'
                }
            },
            axisLabel: {
                color: '#ffffff'
            },
            splitLine: {
                lineStyle: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            }
        },
        series: series,
        backgroundColor: 'transparent'
    };
    
    // 应用配置
    chart.setOption(option);
    
    // 响应式调整
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 新增的R²分数变化曲线图
function createR2LinesChart(containerId, detailedResults, epochs, colors) {
    const chartContainer = document.getElementById(containerId);
    if (!chartContainer) return;
    
    // 初始化图表实例
    const chart = echarts.init(chartContainer);
    
    // 提取数据
    const modelNames = Object.keys(detailedResults);
    const xAxisData = Array.from({length: epochs}, (_, i) => i + 1);
    
    // 准备系列数据
    const series = modelNames.map(model => ({
        name: model,
        type: 'line',
        data: detailedResults[model].r2_scores,
        symbol: 'circle',
        symbolSize: 5,
        lineStyle: {
            width: 2,
            color: colors[model]
        },
        itemStyle: {
            color: colors[model]
        },
        areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                {offset: 0, color: echarts.color.modifyAlpha(colors[model], 0.6)},
                {offset: 1, color: echarts.color.modifyAlpha(colors[model], 0.1)}
            ])
        }
    }));
    
    // 图表配置
    const option = {
        title: {
            text: '训练过程中的R²分数变化',
            left: 'center',
            textStyle: {
                color: '#ffffff'
            }
        },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                let result = `<div>第 ${params[0].axisValue} 轮</div>`;
                params.forEach(param => {
                    result += `<div style="margin: 5px 0;">
                        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background-color:${param.color};"></span>
                        <span style="margin-left:5px">${param.seriesName}: ${param.value.toFixed(6)}</span>
                    </div>`;
                });
                return result;
            }
        },
        legend: {
            data: modelNames,
            textStyle: {
                color: '#ffffff'
            },
            top: 30
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: xAxisData,
            name: 'Epoch',
            nameLocation: 'middle',
            nameGap: 30,
            nameTextStyle: {
                color: '#ffffff'
            },
            axisLine: {
                lineStyle: {
                    color: '#ffffff'
                }
            },
            axisLabel: {
                color: '#ffffff'
            }
        },
        yAxis: {
            type: 'value',
            name: 'R²分数',
            nameTextStyle: {
                color: '#ffffff'
            },
            axisLine: {
                lineStyle: {
                    color: '#ffffff'
                }
            },
            axisLabel: {
                color: '#ffffff'
            },
            splitLine: {
                lineStyle: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            },
            min: function(value) {
                // 确保显示的最小值不高于数据中的最小值，向下取整
                return Math.floor(Math.min(0, value.min * 1.1));
            },
            max: 1.0 // R²分数的理想最大值为1.0
        },
        series: series,
        backgroundColor: 'transparent'
    };
    
    // 应用配置
    chart.setOption(option);
    
    // 响应式调整
    window.addEventListener('resize', function() {
        chart.resize();
    });
} 