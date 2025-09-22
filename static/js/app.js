// Portfolio Management Frontend Application
class PortfolioApp {
    constructor() {
        this.currentSection = 'dashboard';
        this.charts = {};
        this.monitoringInterval = null;
        this.websocket = null;
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupModals();
        this.setupCharts();
        this.loadInitialData();
        this.setupEventListeners();
        this.setupWebSocket();
    }

    setupNavigation() {
        const navToggle = document.getElementById('nav-toggle');
        const navMenu = document.getElementById('nav-menu');
        const navLinks = document.querySelectorAll('.nav-link');

        // Mobile menu toggle
        navToggle?.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });

        // Navigation link handling
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetSection = link.getAttribute('href').substring(1);
                this.showSection(targetSection);

                // Update active nav link
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');

                // Close mobile menu
                navMenu.classList.remove('active');
                navToggle.classList.remove('active');
            });
        });
    }

    showSection(sectionName) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });

        // Show target section
        const targetSection = document.getElementById(sectionName);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = sectionName;

            // Trigger section-specific updates
            this.onSectionChange(sectionName);
        }
    }

    onSectionChange(sectionName) {
        switch (sectionName) {
            case 'dashboard':
                this.updateDashboard();
                break;
            case 'portfolio':
                this.updatePortfolio();
                break;
            case 'risk':
                this.updateRiskAnalysis();
                break;
            case 'monitoring':
                this.updateMonitoring();
                break;
        }
    }

    setupModals() {
        const addPositionModal = document.getElementById('add-position-modal');
        const addPositionBtn = document.getElementById('add-position-btn');
        const closeBtn = addPositionModal?.querySelector('.close');
        const cancelBtn = document.getElementById('cancel-add');
        const addPositionForm = document.getElementById('add-position-form');

        // Open modal
        addPositionBtn?.addEventListener('click', () => {
            addPositionModal.style.display = 'block';
        });

        // Close modal
        const closeModal = () => {
            addPositionModal.style.display = 'none';
            addPositionForm?.reset();
        };

        closeBtn?.addEventListener('click', closeModal);
        cancelBtn?.addEventListener('click', closeModal);

        // Close modal when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target === addPositionModal) {
                closeModal();
            }
        });

        // Handle form submission
        addPositionForm?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.addPosition();
        });
    }

    setupCharts() {
        // Performance Chart
        const perfCtx = document.getElementById('performance-chart');
        if (perfCtx) {
            this.charts.performance = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: this.generateTimeLabels(30),
                    datasets: [{
                        label: 'Portfolio Value',
                        data: this.generateRandomData(30, 120000, 130000),
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                }
                            }
                        }
                    }
                }
            });
        }

        // Risk Chart
        const riskCtx = document.getElementById('risk-chart');
        if (riskCtx) {
            this.charts.risk = new Chart(riskCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Stocks', 'Options', 'Cash'],
                    datasets: [{
                        data: [60, 30, 10],
                        backgroundColor: [
                            '#2563eb',
                            '#10b981',
                            '#f59e0b'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }

        // VaR Backtesting Chart
        const varCtx = document.getElementById('var-backtest-chart');
        if (varCtx) {
            this.charts.varBacktest = new Chart(varCtx, {
                type: 'line',
                data: {
                    labels: this.generateTimeLabels(100),
                    datasets: [{
                        label: 'Daily P&L',
                        data: this.generateRandomData(100, -5000, 8000),
                        borderColor: '#64748b',
                        backgroundColor: 'rgba(100, 116, 139, 0.1)',
                        fill: false
                    }, {
                        label: 'VaR (95%)',
                        data: Array(100).fill(-3245.80),
                        borderColor: '#ef4444',
                        borderDash: [5, 5],
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    setupEventListeners() {
        // Refresh buttons
        document.getElementById('refresh-risk-btn')?.addEventListener('click', () => {
            this.updateRiskAnalysis();
        });

        // Monitoring controls
        document.getElementById('start-monitoring-btn')?.addEventListener('click', () => {
            this.startMonitoring();
        });

        document.getElementById('stop-monitoring-btn')?.addEventListener('click', () => {
            this.stopMonitoring();
        });
    }

    setupWebSocket() {
        // In a real application, this would connect to your Python backend
        // For demo purposes, we'll simulate WebSocket updates
        this.simulateRealTimeUpdates();
    }

    simulateRealTimeUpdates() {
        setInterval(() => {
            if (this.currentSection === 'dashboard') {
                this.updateMetrics();
            }
            if (this.currentSection === 'monitoring') {
                this.addLogEntry();
            }
        }, 5000);
    }

    generateTimeLabels(count) {
        const labels = [];
        const now = new Date();
        for (let i = count - 1; i >= 0; i--) {
            const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
            labels.push(date.toLocaleDateString());
        }
        return labels;
    }

    generateRandomData(count, min, max) {
        const data = [];
        for (let i = 0; i < count; i++) {
            data.push(Math.random() * (max - min) + min);
        }
        return data;
    }

    loadInitialData() {
        this.updateDashboard();
        this.updatePortfolio();
        this.updateRiskAnalysis();
        this.updateMonitoring();
    }

    async updateDashboard() {
        // Simulate API call to get dashboard data
        const data = await this.fetchDashboardData();

        // Update metrics
        document.getElementById('portfolio-value').textContent = '$' + data.portfolioValue.toLocaleString();
        document.getElementById('var-value').textContent = '$' + data.var.toLocaleString();
        document.getElementById('delta-value').textContent = data.delta.toFixed(2);
        document.getElementById('gamma-value').textContent = data.gamma.toFixed(2);
    }

    async updatePortfolio() {
        const positions = await this.fetchPositions();
        const tbody = document.getElementById('positions-tbody');

        if (tbody) {
            tbody.innerHTML = '';
            positions.forEach(position => {
                const row = this.createPositionRow(position);
                tbody.appendChild(row);
            });
        }
    }

    async updateRiskAnalysis() {
        // Update VaR values and Greeks
        const riskData = await this.fetchRiskData();
        // Update UI with risk data
    }

    async updateMonitoring() {
        this.updateHealthIndicators();
        this.updateAlerts();
        this.updatePerformanceMetrics();
    }

    updateHealthIndicators() {
        // Simulate health status updates
        const indicators = document.querySelectorAll('.health-item .status-dot');
        indicators.forEach(dot => {
            // Randomly change status for demo
            if (Math.random() > 0.95) {
                dot.className = 'status-dot offline';
                dot.nextElementSibling.textContent = 'Offline';
            } else {
                dot.className = 'status-dot online';
                dot.nextElementSibling.textContent = 'Online';
            }
        });
    }

    updateAlerts() {
        const alertsList = document.getElementById('alerts-list');
        if (!alertsList) return;

        // Clear existing alerts
        alertsList.innerHTML = '';

        // Add sample alerts
        const alerts = [
            'Delta limit approaching threshold',
            'MSTR-BTC correlation spike detected',
            'High volatility in portfolio detected'
        ];

        alerts.forEach(alert => {
            const alertDiv = document.createElement('div');
            alertDiv.className = 'alert-item';
            alertDiv.textContent = alert;
            alertsList.appendChild(alertDiv);
        });
    }

    updatePerformanceMetrics() {
        // Update performance metrics with random values for demo
        const metrics = {
            'Update Frequency': Math.floor(Math.random() * 5 + 8) + 's',
            'Processing Time': Math.floor(Math.random() * 10 + 10) + 'ms',
            'Cache Hit Rate': (Math.random() * 5 + 90).toFixed(1) + '%'
        };

        document.querySelectorAll('.perf-item').forEach((item, index) => {
            const key = Object.keys(metrics)[index];
            if (key) {
                item.lastElementChild.textContent = metrics[key];
            }
        });
    }

    addLogEntry() {
        const logsViewer = document.getElementById('logs-viewer');
        if (!logsViewer) return;

        const timestamp = new Date().toLocaleTimeString();
        const messages = [
            'Portfolio Greeks calculated successfully',
            'Risk limits checked - all within bounds',
            'Market data feed updated',
            'Model validation completed',
            'VaR calculation finished'
        ];

        const message = messages[Math.floor(Math.random() * messages.length)];
        const logEntry = document.createElement('div');
        logEntry.textContent = `[${timestamp}] ${message}`;

        logsViewer.appendChild(logEntry);
        logsViewer.scrollTop = logsViewer.scrollHeight;

        // Keep only last 50 entries
        while (logsViewer.children.length > 50) {
            logsViewer.removeChild(logsViewer.firstChild);
        }
    }

    updateMetrics() {
        // Simulate real-time metric updates
        const portfolioValue = 125420.50 + (Math.random() - 0.5) * 1000;
        const var95 = 3245.80 + (Math.random() - 0.5) * 200;
        const delta = 0.42 + (Math.random() - 0.5) * 0.1;
        const gamma = 0.12 + (Math.random() - 0.5) * 0.02;

        document.getElementById('portfolio-value').textContent = '$' + portfolioValue.toLocaleString();
        document.getElementById('var-value').textContent = '$' + var95.toLocaleString();
        document.getElementById('delta-value').textContent = delta.toFixed(2);
        document.getElementById('gamma-value').textContent = gamma.toFixed(2);
    }

    createPositionRow(position) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${position.symbol}</td>
            <td>${position.type}</td>
            <td>${position.quantity}</td>
            <td>$${position.currentPrice.toFixed(2)}</td>
            <td>$${position.marketValue.toLocaleString()}</td>
            <td class="${position.pnl >= 0 ? 'text-success' : 'text-danger'}">
                $${position.pnl.toLocaleString()}
            </td>
            <td>${position.delta.toFixed(3)}</td>
            <td>${position.gamma.toFixed(3)}</td>
            <td>
                <button class="btn btn-sm btn-secondary" onclick="app.editPosition('${position.id}')">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="app.deletePosition('${position.id}')">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        `;
        return row;
    }

    async addPosition() {
        const form = document.getElementById('add-position-form');
        const formData = new FormData(form);

        const position = {
            symbol: formData.get('symbol'),
            type: formData.get('position-type'),
            quantity: parseFloat(formData.get('quantity')),
            price: parseFloat(formData.get('price'))
        };

        try {
            // In a real app, this would make an API call to your Python backend
            await this.submitPosition(position);

            // Close modal and refresh portfolio
            document.getElementById('add-position-modal').style.display = 'none';
            form.reset();
            this.updatePortfolio();

            this.showNotification('Position added successfully', 'success');
        } catch (error) {
            this.showNotification('Error adding position', 'error');
        }
    }

    async editPosition(id) {
        // Implement position editing
        console.log('Edit position:', id);
    }

    async deletePosition(id) {
        if (confirm('Are you sure you want to delete this position?')) {
            try {
                await this.removePosition(id);
                this.updatePortfolio();
                this.showNotification('Position deleted successfully', 'success');
            } catch (error) {
                this.showNotification('Error deleting position', 'error');
            }
        }
    }

    startMonitoring() {
        if (this.monitoringInterval) return;

        this.monitoringInterval = setInterval(() => {
            this.updateRealTimeData();
        }, 10000); // Update every 10 seconds

        this.showNotification('Real-time monitoring started', 'success');
    }

    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }

        this.showNotification('Real-time monitoring stopped', 'info');
    }

    updateRealTimeData() {
        // Update all real-time data
        this.updateMetrics();
        this.updateHealthIndicators();
        this.addLogEntry();
    }

    showNotification(message, type = 'info') {
        // Create and show notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    // Mock API functions - in a real app, these would call your Python backend
    async fetchDashboardData() {
        return {
            portfolioValue: 125420.50,
            var: 3245.80,
            delta: 0.42,
            gamma: 0.12
        };
    }

    async fetchPositions() {
        return [
            {
                id: '1',
                symbol: 'MSTR',
                type: 'Stock',
                quantity: 100,
                currentPrice: 850.25,
                marketValue: 85025,
                pnl: 5420,
                delta: 0.85,
                gamma: 0.02
            },
            {
                id: '2',
                symbol: 'MSTR 850C',
                type: 'Call',
                quantity: 10,
                currentPrice: 25.50,
                marketValue: 25500,
                pnl: -1200,
                delta: 0.65,
                gamma: 0.15
            },
            {
                id: '3',
                symbol: 'BTC-USD',
                type: 'Crypto',
                quantity: 0.5,
                currentPrice: 65420.30,
                marketValue: 32710,
                pnl: 2150,
                delta: 1.0,
                gamma: 0.0
            }
        ];
    }

    async fetchRiskData() {
        return {
            var95_1d: 3245.80,
            var99_1d: 4892.15,
            var95_10d: 10267.34,
            delta: 0.42,
            gamma: 0.12,
            vega: 45.8,
            theta: -12.3
        };
    }

    async submitPosition(position) {
        // Mock API call - replace with actual endpoint
        return new Promise((resolve) => {
            setTimeout(resolve, 1000);
        });
    }

    async removePosition(id) {
        // Mock API call - replace with actual endpoint
        return new Promise((resolve) => {
            setTimeout(resolve, 500);
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PortfolioApp();
});

// Add notification styles
const notificationStyles = `
    .notification {
        position: fixed;
        top: 100px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: 500;
        z-index: 3000;
        animation: slideIn 0.3s ease;
    }

    .notification-success {
        background-color: var(--success-color);
    }

    .notification-error {
        background-color: var(--danger-color);
    }

    .notification-info {
        background-color: var(--primary-color);
    }

    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;

// Add styles to head
const styleElement = document.createElement('style');
styleElement.textContent = notificationStyles;
document.head.appendChild(styleElement);