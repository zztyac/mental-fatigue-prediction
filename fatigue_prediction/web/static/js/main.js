/**
 * 金属多轴疲劳寿命预测系统 - 主JS文件
 */

// 等待DOM加载完成
document.addEventListener('DOMContentLoaded', function() {
    // 初始化工具提示
    initTooltips();
    
    // 初始化动画元素
    initAnimations();
    
    // 添加表单验证
    initFormValidation();
    
    // 初始化图表响应式调整
    initChartResize();
    
    // 初始化导航栏效果
    initNavbarEffects();
    
    // 添加卡片悬停效果
    initCardHoverEffects();
    
    // 初始化交互式元素
    initInteractiveElements();
    
    // 控制台欢迎信息
    console.log(
        "%c金属多轴疲劳寿命预测系统 %c基于深度学习的金属疲劳寿命预测",
        "color: #fff; background: #2c3e50; padding: 5px; border-radius: 3px 0 0 3px;",
        "color: #fff; background: #3498db; padding: 5px; border-radius: 0 3px 3px 0;"
    );
});

/**
 * 初始化Bootstrap工具提示
 */
function initTooltips() {
    try {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl, {
                animation: true,
                delay: { show: 100, hide: 100 }
            });
        });
    } catch (e) {
        console.warn("工具提示初始化失败:", e);
    }
}

/**
 * 初始化动画元素
 */
function initAnimations() {
    // 为没有AOS属性的卡片添加淡入效果
    document.querySelectorAll('.card:not([data-aos])').forEach(function(card, index) {
        card.setAttribute('data-aos', 'fade-up');
        card.setAttribute('data-aos-delay', (index * 50).toString());
        card.setAttribute('data-aos-duration', '800');
    });
    
    // 为按钮添加悬停动画
    document.querySelectorAll('.btn').forEach(function(btn) {
        btn.addEventListener('mouseover', function() {
            this.classList.add('animate__animated', 'animate__pulse');
        });
        
        btn.addEventListener('animationend', function() {
            this.classList.remove('animate__animated', 'animate__pulse');
        });
    });
}

/**
 * 初始化导航栏效果
 */
function initNavbarEffects() {
    const navbar = document.querySelector('.navbar');
    
    if (navbar) {
        // 页面加载时先检查滚动位置
        if (window.scrollY > 50) {
            navbar.classList.add('navbar-scrolled');
        }
        
        // 滚动时改变导航栏样式
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                navbar.classList.add('navbar-scrolled');
                
                // 滚动时为导航项添加过渡延迟
                document.querySelectorAll('.navbar-nav .nav-link').forEach(function(link, index) {
                    link.style.transitionDelay = (index * 0.05) + 's';
                });
            } else {
                navbar.classList.remove('navbar-scrolled');
                
                // 重置过渡延迟
                document.querySelectorAll('.navbar-nav .nav-link').forEach(function(link) {
                    link.style.transitionDelay = '0s';
                });
            }
        });
        
        // 激活的导航项添加特效
        const activeNavItem = navbar.querySelector('.nav-link.active');
        if (activeNavItem) {
            activeNavItem.classList.add('nav-link-active');
            
            // 为激活的导航项添加动画
            const indicator = document.createElement('span');
            indicator.classList.add('nav-indicator');
            activeNavItem.appendChild(indicator);
        }
        
        // 为所有导航链接添加悬停效果
        document.querySelectorAll('.navbar-nav .nav-link').forEach(function(link) {
            link.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-3px)';
            });
            
            link.addEventListener('mouseleave', function() {
                this.style.transform = '';
            });
        });
    }
}

/**
 * 添加卡片悬停效果
 */
function initCardHoverEffects() {
    document.querySelectorAll('.card').forEach(function(card) {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px)';
            this.style.boxShadow = '0 15px 30px rgba(0,0,0,0.1)';
            this.style.transition = 'all 0.3s ease';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });
}

/**
 * 初始化交互式元素
 */
function initInteractiveElements() {
    // 添加进度条动画
    document.querySelectorAll('.progress-bar').forEach(function(bar) {
        const targetWidth = bar.style.width || bar.getAttribute('aria-valuenow') + '%';
        
        // 重置宽度为0
        bar.style.width = '0%';
        
        // 使用setTimeout以确保重置被应用
        setTimeout(function() {
            bar.style.width = targetWidth;
            bar.style.transition = 'width 1s ease-in-out';
        }, 100);
    });
    
    // 添加数值计数动画
    document.querySelectorAll('.counting-number').forEach(function(element) {
        const target = parseInt(element.getAttribute('data-target'));
        const duration = 2000; // ms
        const frameDuration = 1000/60; // 60fps
        const totalFrames = Math.round(duration / frameDuration);
        let frame = 0;
        
        const counter = setInterval(function() {
            frame++;
            const progress = frame / totalFrames;
            const currentCount = Math.round(target * progress);
            
            if (currentCount >= target) {
                element.textContent = target;
                clearInterval(counter);
            } else {
                element.textContent = currentCount;
            }
        }, frameDuration);
    });
}

/**
 * 初始化表单验证
 */
function initFormValidation() {
    // 获取所有需要验证的表单
    var forms = document.querySelectorAll('.needs-validation');
    
    // 循环处理表单
    Array.from(forms).forEach(function(form) {
        // 实时验证
        form.querySelectorAll('input, select, textarea').forEach(function(input) {
            input.addEventListener('input', function() {
                if (this.checkValidity()) {
                    this.classList.remove('is-invalid');
                    this.classList.add('is-valid');
                } else {
                    this.classList.remove('is-valid');
                    this.classList.add('is-invalid');
                }
            });
        });
        
        // 提交验证
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                
                // 显示所有无效输入的反馈
                form.querySelectorAll(':invalid').forEach(function(input) {
                    input.classList.add('is-invalid');
                });
                
                // 滚动到第一个无效输入
                form.querySelector(':invalid').scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });
                
                // 显示提交错误通知
                showNotification('请检查表单中的错误并重试', 'warning');
            }
            
            form.classList.add('was-validated');
        }, false);
    });
}

/**
 * 初始化图表响应式调整
 */
function initChartResize() {
    // 简单的节流函数
    function throttle(func, delay) {
        let timeoutId;
        return function() {
            if (!timeoutId) {
                timeoutId = setTimeout(() => {
                    func.apply(this, arguments);
                    timeoutId = null;
                }, delay);
            }
        };
    }
    
    // 调整所有图表大小
    const resizeCharts = throttle(function() {
        if (window.echarts) {
            const charts = document.querySelectorAll('.chart-container, [id$="-chart"]');
            charts.forEach(function(container) {
                const chart = window.echarts.getInstanceByDom(container);
                if (chart) {
                    chart.resize();
                }
            });
        }
    }, 100);
    
    // 监听窗口大小变化
    window.addEventListener('resize', resizeCharts);
    
    // 监听标签页切换，因为这可能影响图表容器的可见性
    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(function(tab) {
        tab.addEventListener('shown.bs.tab', resizeCharts);
    });
}

/**
 * 格式化时间戳为可读字符串
 * @param {number} timestamp - 时间戳(毫秒)
 * @returns {string} 格式化后的时间字符串
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return '未知时间';
    
    try {
        // 使用Moment.js如果可用
        if (typeof moment !== 'undefined') {
            return moment(timestamp).format('YYYY-MM-DD HH:mm:ss');
        }
        
        // 回退到原生Date
        var date = new Date(timestamp);
        var year = date.getFullYear();
        var month = ('0' + (date.getMonth() + 1)).slice(-2);
        var day = ('0' + date.getDate()).slice(-2);
        var hours = ('0' + date.getHours()).slice(-2);
        var minutes = ('0' + date.getMinutes()).slice(-2);
        var seconds = ('0' + date.getSeconds()).slice(-2);
        
        return year + '-' + month + '-' + day + ' ' + hours + ':' + minutes + ':' + seconds;
    } catch (e) {
        console.error('格式化时间戳出错:', e);
        return '时间格式错误';
    }
}

/**
 * 显示通知消息
 * @param {string} message - 消息内容
 * @param {string} type - 消息类型 (success, warning, danger, info)
 * @param {number} duration - 显示时长(毫秒)
 */
function showNotification(message, type = 'info', duration = 3000) {
    // 创建通知元素
    var notification = document.createElement('div');
    notification.className = 'alert alert-' + type + ' alert-dismissible fade notification shadow';
    
    // 为不同类型设置不同图标
    var icon = 'info-circle';
    if (type === 'success') icon = 'check-circle';
    else if (type === 'warning') icon = 'exclamation-triangle';
    else if (type === 'danger') icon = 'exclamation-circle';
    
    notification.innerHTML = 
        '<div class="d-flex align-items-center">' +
            '<i class="fas fa-' + icon + ' me-2"></i>' +
            '<div>' + message + '</div>' +
        '</div>' +
        '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>';
    
    // 设置样式
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '1050';
    notification.style.minWidth = '300px';
    notification.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.2)';
    notification.style.opacity = '0';
    notification.style.transform = 'translateY(-20px)';
    notification.style.transition = 'all 0.3s ease';
    
    // 添加到body
    document.body.appendChild(notification);
    
    // 显示通知
    setTimeout(function() {
        notification.style.opacity = '1';
        notification.style.transform = 'translateY(0)';
    }, 10);
    
    // 自动关闭
    setTimeout(function() {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-20px)';
        
        // 移除元素
        setTimeout(function() {
            notification.remove();
        }, 300);
    }, duration);
}

/**
 * 确认对话框
 * @param {string} message - 确认消息
 * @param {object} options - 选项配置
 * @param {function} callback - 确认后的回调函数
 */
function confirmAction(message, options = {}, callback) {
    // 默认选项
    const defaultOptions = {
        title: '确认操作',
        confirmText: '确定',
        cancelText: '取消',
        confirmButtonClass: 'btn-primary',
        iconClass: 'fa-question-circle'
    };
    
    // 合并选项
    const settings = {...defaultOptions, ...options};
    
    // 创建模态框
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'confirmModal';
    modal.tabIndex = '-1';
    modal.setAttribute('aria-labelledby', 'confirmModalLabel');
    modal.setAttribute('aria-hidden', 'true');
    
    modal.innerHTML = `
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="confirmModalLabel">${settings.title}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div class="d-flex align-items-center">
                        <i class="fas ${settings.iconClass} fa-2x me-3 text-${settings.confirmButtonClass.replace('btn-', '')}"></i>
                        <div>${message}</div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">${settings.cancelText}</button>
                    <button type="button" class="btn ${settings.confirmButtonClass}" id="confirmActionBtn">${settings.confirmText}</button>
                </div>
            </div>
        </div>
    `;
    
    // 添加到body
    document.body.appendChild(modal);
    
    // 创建Bootstrap模态对象
    var confirmModal = new bootstrap.Modal(modal);
    
    // 显示模态框
    confirmModal.show();
    
    // 绑定确认按钮事件
    document.getElementById('confirmActionBtn').addEventListener('click', function() {
        confirmModal.hide();
        callback();
        
        // 在隐藏后删除模态框
        modal.addEventListener('hidden.bs.modal', function() {
            modal.remove();
        });
    });
    
    // 在隐藏后删除模态框
    modal.addEventListener('hidden.bs.modal', function() {
        setTimeout(() => {
            modal.remove();
        }, 300);
    });
}

// 添加一些自定义CSS规则
function addCustomStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .navbar-scrolled {
            background-color: rgba(245, 247, 250, 0.85) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1) !important;
        }
        
        .navbar-scrolled .navbar-brand,
        .navbar-scrolled .navbar-nav .nav-link {
            color: var(--primary-color);
        }
        
        .navbar-scrolled .navbar-toggler-icon {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 30 30'%3e%3cpath stroke='rgba(44, 62, 80, 1)' stroke-linecap='round' stroke-miterlimit='10' stroke-width='2' d='M4 7h22M4 15h22M4 23h22'/%3e%3c/svg%3e");
        }
        
        .nav-link-active {
            position: relative;
        }
        
        .nav-indicator {
            position: absolute;
            bottom: -3px;
            left: 50%;
            transform: translateX(-50%);
            width: 6px;
            height: 6px;
            background-color: var(--accent-color);
            border-radius: 50%;
            transition: all 0.3s ease;
        }
        
        .nav-link:hover .nav-indicator {
            width: 20px;
            border-radius: 3px;
        }
        
        @keyframes navItemFade {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .navbar-nav .nav-item {
            animation: navItemFade 0.5s forwards;
        }
        
        .page-loader {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            z-index: 9999;
            background-color: transparent;
            transition: opacity 0.5s ease;
        }
        
        .page-loader .progress {
            height: 3px;
            background-color: transparent;
        }
        
        #back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 1000;
        }
        
        #back-to-top.show {
            opacity: 1;
            visibility: visible;
        }
        
        .notification {
            max-width: 400px;
        }
    `;
    document.head.appendChild(style);
}

// 页面加载时添加自定义样式
addCustomStyles(); 