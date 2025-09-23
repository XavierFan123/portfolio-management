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
        if (navToggle && navMenu) {
            navToggle.addEventListener('click', () => {
                navMenu.classList.toggle('active');
                navToggle.classList.toggle('active');
            });
        }

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
                if (navMenu && navToggle) {
                    navMenu.classList.remove('active');
                    navToggle.classList.remove('active');
                }
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
        this.setupAddPositionModal();
        this.setupEditPositionModal();
    }

    setupAddPositionModal() {
        const addPositionModal = document.getElementById('add-position-modal');
        const addPositionBtn = document.getElementById('add-position-btn');
        const closeBtn = addPositionModal?.querySelector('.close');
        const cancelBtn = document.getElementById('cancel-add');
        const addPositionForm = document.getElementById('add-position-form');
        const positionTypeSelect = document.getElementById('position-type');

        // Open modal
        if (addPositionBtn && addPositionModal) {
            addPositionBtn.addEventListener('click', () => {
                addPositionModal.style.display = 'block';
            });
        }

        // Handle position type change
        if (positionTypeSelect) {
            positionTypeSelect.addEventListener('change', (e) => {
                this.toggleFormFields(e.target.value);
            });
        }

        // Close modal
        const closeModal = () => {
            if (addPositionModal) {
                addPositionModal.style.display = 'none';
            }
            if (addPositionForm) {
                addPositionForm.reset();
                this.toggleFormFields(''); // Reset form fields
            }
        };

        if (closeBtn) {
            closeBtn.addEventListener('click', closeModal);
        }
        if (cancelBtn) {
            cancelBtn.addEventListener('click', closeModal);
        }

        // Close modal when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target === addPositionModal) {
                closeModal();
            }
        });

        // Handle form submission
        if (addPositionForm) {
            addPositionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.addPosition();
            });
        }
    }

    setupEditPositionModal() {
        const editPositionModal = document.getElementById('edit-position-modal');
        const closeBtn = editPositionModal?.querySelector('.close');
        const cancelBtn = document.getElementById('cancel-edit');
        const editPositionForm = document.getElementById('edit-position-form');

        // Close modal
        const closeModal = () => {
            if (editPositionModal) {
                editPositionModal.style.display = 'none';
            }
            if (editPositionForm) {
                editPositionForm.reset();
            }
        };

        if (closeBtn) {
            closeBtn.addEventListener('click', closeModal);
        }
        if (cancelBtn) {
            cancelBtn.addEventListener('click', closeModal);
        }

        // Close modal when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target === editPositionModal) {
                closeModal();
            }
        });

        // Handle form submission
        if (editPositionForm) {
            editPositionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.updatePosition();
            });
        }
    }

    toggleFormFields(positionType) {
        const avgCostGroup = document.getElementById('avg-cost-group');
        const strikePriceGroup = document.getElementById('strike-price-group');
        const expirationGroup = document.getElementById('expiration-group');
        const optionStyleGroup = document.getElementById('option-style-group');
        const premiumGroup = document.getElementById('premium-group');

        const strikePriceInput = document.getElementById('strike-price');
        const expirationInput = document.getElementById('expiration-date');
        const optionStyleInput = document.getElementById('option-style');

        // Hide all optional fields first
        if (avgCostGroup) avgCostGroup.style.display = 'none';
        if (strikePriceGroup) strikePriceGroup.style.display = 'none';
        if (expirationGroup) expirationGroup.style.display = 'none';
        if (optionStyleGroup) optionStyleGroup.style.display = 'none';
        if (premiumGroup) premiumGroup.style.display = 'none';

        // Remove required attribute from all optional fields
        if (strikePriceInput) strikePriceInput.removeAttribute('required');
        if (expirationInput) expirationInput.removeAttribute('required');
        if (optionStyleInput) optionStyleInput.removeAttribute('required');

        // Show relevant fields based on position type
        if (positionType === 'stock' || positionType === 'crypto') {
            if (avgCostGroup) avgCostGroup.style.display = 'block';
        } else if (positionType === 'call' || positionType === 'put') {
            // Show all option-specific fields
            if (strikePriceGroup) strikePriceGroup.style.display = 'block';
            if (expirationGroup) expirationGroup.style.display = 'block';
            if (optionStyleGroup) optionStyleGroup.style.display = 'block';
            if (premiumGroup) premiumGroup.style.display = 'block';

            // Make option fields required
            if (strikePriceInput) strikePriceInput.setAttribute('required', 'required');
            if (expirationInput) expirationInput.setAttribute('required', 'required');
            if (optionStyleInput) optionStyleInput.setAttribute('required', 'required');
        }
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
                    aspectRatio: 2,
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
                    aspectRatio: 1,
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
                    aspectRatio: 2,
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
        const refreshRiskBtn = document.getElementById('refresh-risk-btn');
        if (refreshRiskBtn) {
            refreshRiskBtn.addEventListener('click', () => {
                this.updateRiskAnalysis();
            });
        }

        // Monitoring controls
        const startMonitoringBtn = document.getElementById('start-monitoring-btn');
        const stopMonitoringBtn = document.getElementById('stop-monitoring-btn');

        if (startMonitoringBtn) {
            startMonitoringBtn.addEventListener('click', () => {
                this.startMonitoring();
            });
        }

        if (stopMonitoringBtn) {
            stopMonitoringBtn.addEventListener('click', () => {
                this.stopMonitoring();
            });
        }
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
        try {
            // Simulate API call to get dashboard data
            const response = await fetch('/api/dashboard');
            const data = await response.json();

            if (data.status === 'success') {
                // Update metrics
                const portfolioValueEl = document.getElementById('portfolio-value');
                const varValueEl = document.getElementById('var-value');
                const deltaValueEl = document.getElementById('delta-value');
                const gammaValueEl = document.getElementById('gamma-value');

                if (portfolioValueEl) portfolioValueEl.textContent = '$' + data.portfolioValue.toLocaleString();
                if (varValueEl) varValueEl.textContent = '$' + data.var.toLocaleString();
                if (deltaValueEl) deltaValueEl.textContent = data.delta.toFixed(2);
                if (gammaValueEl) gammaValueEl.textContent = data.gamma.toFixed(2);
            }
        } catch (error) {
            console.log('Using demo data for dashboard');
            // Use demo data if API is not available
            const portfolioValueEl = document.getElementById('portfolio-value');
            const varValueEl = document.getElementById('var-value');
            const deltaValueEl = document.getElementById('delta-value');
            const gammaValueEl = document.getElementById('gamma-value');

            if (portfolioValueEl) portfolioValueEl.textContent = '$125,420.50';
            if (varValueEl) varValueEl.textContent = '$3,245.80';
            if (deltaValueEl) deltaValueEl.textContent = '0.42';
            if (gammaValueEl) gammaValueEl.textContent = '0.12';
        }
    }

    async updatePortfolio() {
        try {
            const response = await fetch('/api/positions');
            const data = await response.json();

            if (data.status === 'success') {
                const tbody = document.getElementById('positions-tbody');
                if (tbody) {
                    tbody.innerHTML = '';
                    data.positions.forEach(position => {
                        const row = this.createPositionRow(position);
                        tbody.appendChild(row);
                    });
                }
            }
        } catch (error) {
            console.log('Using demo data for portfolio');
            // Use demo data if API is not available
            const positions = [
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
                }
            ];

            const tbody = document.getElementById('positions-tbody');
            if (tbody) {
                tbody.innerHTML = '';
                positions.forEach(position => {
                    const row = this.createPositionRow(position);
                    tbody.appendChild(row);
                });
            }
        }
    }

    async updateRiskAnalysis() {
        // Update VaR values and Greeks
        try {
            const response = await fetch('/api/risk');
            const data = await response.json();

            if (data.status === 'success') {
                // Update VaR displays
                const varItems = document.querySelectorAll('.var-item .var-value');
                if (varItems.length >= 3) {
                    varItems[0].textContent = '$' + data.var95_1d.toLocaleString();
                    varItems[1].textContent = '$' + data.var99_1d.toLocaleString();
                    varItems[2].textContent = '$' + data.var95_10d.toLocaleString();
                }

                // Update Greeks
                const greekItems = document.querySelectorAll('.greek-item .greek-value');
                if (greekItems.length >= 4) {
                    greekItems[0].textContent = data.delta.toFixed(2);
                    greekItems[1].textContent = data.gamma.toFixed(2);
                    greekItems[2].textContent = data.vega.toFixed(1);
                    greekItems[3].textContent = data.theta.toFixed(1);
                }
            }
        } catch (error) {
            console.log('Using demo data for risk analysis');
        }
    }

    async updateMonitoring() {
        this.updateHealthIndicators();
        this.updateAlerts();
        this.updatePerformanceMetrics();
    }

    updateHealthIndicators() {
        // Simulate health status updates
        const healthItems = document.querySelectorAll('.health-item');
        healthItems.forEach(item => {
            const statusIndicator = item.querySelector('.status-indicator');
            if (statusIndicator) {
                const dot = statusIndicator.querySelector('.status-dot');
                const textSpan = statusIndicator.querySelector('span:last-child');

                if (dot && textSpan) {
                    // Randomly change status for demo
                    if (Math.random() > 0.95) {
                        dot.className = 'status-dot offline';
                        textSpan.textContent = 'Offline';
                    } else {
                        dot.className = 'status-dot online';
                        textSpan.textContent = 'Online';
                    }
                }
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
        const perfItems = document.querySelectorAll('.perf-item');
        const metrics = [
            Math.floor(Math.random() * 5 + 8) + 's',
            Math.floor(Math.random() * 10 + 10) + 'ms',
            (Math.random() * 5 + 90).toFixed(1) + '%'
        ];

        perfItems.forEach((item, index) => {
            if (index < metrics.length) {
                const valueSpan = item.querySelector('span:last-child');
                if (valueSpan) {
                    valueSpan.textContent = metrics[index];
                }
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

        const portfolioValueEl = document.getElementById('portfolio-value');
        const varValueEl = document.getElementById('var-value');
        const deltaValueEl = document.getElementById('delta-value');
        const gammaValueEl = document.getElementById('gamma-value');

        if (portfolioValueEl) portfolioValueEl.textContent = '$' + portfolioValue.toLocaleString();
        if (varValueEl) varValueEl.textContent = '$' + var95.toLocaleString();
        if (deltaValueEl) deltaValueEl.textContent = delta.toFixed(2);
        if (gammaValueEl) gammaValueEl.textContent = gamma.toFixed(2);
    }

    createPositionRow(position) {
        const row = document.createElement('tr');

        // Determine what to show in Strike/Underlying column
        let strikeUnderlyingDisplay = '';
        if (position.type === 'Call' || position.type === 'Put') {
            // For options: show Strike / Underlying Price
            const strikePrice = position.strikePrice ? `$${position.strikePrice.toFixed(2)}` : 'N/A';
            const underlyingPrice = position.underlyingPrice ? `$${position.underlyingPrice.toFixed(2)}` : 'N/A';
            strikeUnderlyingDisplay = `${strikePrice} / ${underlyingPrice}`;
        } else {
            // For stocks/crypto: show dash
            strikeUnderlyingDisplay = '-';
        }

        row.innerHTML = `
            <td>${position.symbol}</td>
            <td>${position.type}</td>
            <td>${position.quantity}</td>
            <td>${strikeUnderlyingDisplay}</td>
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
        if (!form) return;

        const formData = new FormData(form);

        // Build position object based on form data
        const position = {
            symbol: formData.get('symbol'),
            type: formData.get('type'),
            quantity: parseFloat(formData.get('quantity'))
        };

        // Add optional fields based on position type
        const positionType = formData.get('type');

        if (positionType === 'stock' || positionType === 'crypto') {
            const avgCost = formData.get('avg_cost');
            if (avgCost && avgCost.trim() !== '') {
                position.avg_cost = parseFloat(avgCost);
            }
        } else if (positionType === 'call' || positionType === 'put') {
            const strikePrice = formData.get('strike_price');
            const premium = formData.get('premium');

            if (!strikePrice || strikePrice.trim() === '') {
                this.showNotification('Strike price is required for options', 'error');
                return;
            }

            position.strike_price = parseFloat(strikePrice);

            if (premium && premium.trim() !== '') {
                position.premium = parseFloat(premium);
            }
        }

        try {
            const response = await fetch('/api/positions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(position)
            });

            const data = await response.json();

            if (data.status === 'success') {
                // Close modal and refresh portfolio
                const modal = document.getElementById('add-position-modal');
                if (modal) modal.style.display = 'none';
                form.reset();
                this.toggleFormFields(''); // Reset form fields
                this.updatePortfolio();
                this.updateDashboard(); // Update dashboard metrics
                this.showNotification('Position added successfully', 'success');
            } else {
                this.showNotification('Error adding position: ' + (data.error || data.message || 'Unknown error'), 'error');
            }
        } catch (error) {
            console.error('Error adding position:', error);
            this.showNotification('Network error while adding position', 'error');
        }
    }

    async editPosition(id) {
        try {
            // Fetch position data
            const response = await fetch(`/api/positions/${id}`);
            const data = await response.json();

            if (data.status === 'success') {
                const position = data.position;

                // Populate the edit form
                document.getElementById('edit-position-id').value = position.id;
                document.getElementById('edit-symbol').value = position.symbol;
                document.getElementById('edit-position-type').value = position.type;
                document.getElementById('edit-quantity').value = position.quantity;
                document.getElementById('edit-avg-cost').value = position.avg_cost;

                // Handle different position types
                const editAvgCostGroup = document.getElementById('edit-avg-cost-group');
                const editStrikePriceGroup = document.getElementById('edit-strike-price-group');

                if (position.type === 'call' || position.type === 'put') {
                    if (editAvgCostGroup) editAvgCostGroup.style.display = 'none';
                    if (editStrikePriceGroup) {
                        editStrikePriceGroup.style.display = 'block';
                        document.getElementById('edit-strike-price').value = position.strike_price || '';
                    }
                } else {
                    if (editAvgCostGroup) editAvgCostGroup.style.display = 'block';
                    if (editStrikePriceGroup) editStrikePriceGroup.style.display = 'none';
                }

                // Show the modal
                const editModal = document.getElementById('edit-position-modal');
                if (editModal) {
                    editModal.style.display = 'block';
                }
            } else {
                this.showNotification('Error loading position data: ' + data.error, 'error');
            }
        } catch (error) {
            console.error('Error editing position:', error);
            this.showNotification('Network error while loading position data', 'error');
        }
    }

    async updatePosition() {
        const form = document.getElementById('edit-position-form');
        if (!form) return;

        const formData = new FormData(form);
        const positionId = formData.get('position_id');

        // Build update object
        const updates = {
            quantity: parseFloat(formData.get('quantity')),
            avg_cost: parseFloat(formData.get('avg_cost'))
        };

        try {
            const response = await fetch(`/api/positions/${positionId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(updates)
            });

            const data = await response.json();

            if (data.status === 'success') {
                // Close modal and refresh portfolio
                const modal = document.getElementById('edit-position-modal');
                if (modal) modal.style.display = 'none';
                form.reset();
                this.updatePortfolio();
                this.updateDashboard(); // Update dashboard metrics
                this.showNotification('Position updated successfully', 'success');
            } else {
                this.showNotification('Error updating position: ' + (data.error || data.message || 'Unknown error'), 'error');
            }
        } catch (error) {
            console.error('Error updating position:', error);
            this.showNotification('Network error while updating position', 'error');
        }
    }

    async deletePosition(id) {
        if (confirm('Are you sure you want to delete this position?')) {
            try {
                const response = await fetch(`/api/positions/${id}`, {
                    method: 'DELETE'
                });

                const data = await response.json();

                if (data.status === 'success') {
                    this.updatePortfolio();
                    this.showNotification('Position deleted successfully', 'success');
                } else {
                    this.showNotification('Error deleting position: ' + data.error, 'error');
                }
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

    .text-success {
        color: var(--success-color) !important;
    }

    .text-danger {
        color: var(--danger-color) !important;
    }

    .btn-sm {
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        margin: 0 0.125rem;
    }
`;

// Add styles to head
const styleElement = document.createElement('style');
styleElement.textContent = notificationStyles;
document.head.appendChild(styleElement);