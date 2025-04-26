let priceChart = null;

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const symbol = document.getElementById('symbol').value;
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    
    // Show loading, hide result and error
    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
    
    try {
        const formData = new FormData();
        formData.append('symbol', symbol);
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update results
        updateResults(data);
        
        // Update chart
        updateChart(data.performance.dates, data.performance.price_history);
        
        // Update news
        updateNews(data.news);
        
        // Show results
        loadingDiv.classList.add('hidden');
        resultDiv.classList.remove('hidden');
        
    } catch (error) {
        loadingDiv.classList.add('hidden');
        errorDiv.classList.remove('hidden');
        errorDiv.querySelector('.error-message').textContent = error.message;
    }
});

document.getElementById('export-btn').addEventListener('click', async () => {
    const symbol = document.getElementById('symbol').value;
    
    try {
        const formData = new FormData();
        formData.append('symbol', symbol);
        
        const response = await fetch('/export', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${symbol}_historical_data.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        } else {
            throw new Error('Failed to export data');
        }
    } catch (error) {
        alert('Error exporting data: ' + error.message);
    }
});

function updateChart(dates, prices) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    if (priceChart) {
        priceChart.destroy();
    }
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Stock Price',
                data: prices,
                borderColor: '#28a745',
                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

function updateNews(news) {
    const newsContainer = document.getElementById('news-list');
    newsContainer.innerHTML = '';
    
    if (news.length === 0) {
        newsContainer.innerHTML = '<p>No recent news available</p>';
        return;
    }
    
    news.forEach(item => {
        const sentiment = item.sentiment;
        const sentimentClass = sentiment > 0 ? 'positive' : sentiment < 0 ? 'negative' : '';
        
        const newsItem = document.createElement('div');
        newsItem.className = `news-item ${sentimentClass}`;
        
        const newsContent = `
            <h4>${item.title}</h4>
            <p>${item.summary}</p>
            <div class="news-meta">
                <span class="news-source">${item.source}</span>
                <span class="news-date">${new Date(item.time_published).toLocaleString()}</span>
            </div>
            <div class="news-footer">
                ${item.url ? `<a href="${item.url}" target="_blank" class="news-link">Read More</a>` : ''}
            </div>
        `;
        
        newsItem.innerHTML = newsContent;
        newsContainer.appendChild(newsItem);
    });
}

// Add news filter functionality
document.querySelectorAll('.news-filter').forEach(button => {
    button.addEventListener('click', (e) => {
        // Update active button
        document.querySelectorAll('.news-filter').forEach(btn => btn.classList.remove('active'));
        e.target.classList.add('active');
        
        // Filter news items
        const sentiment = e.target.dataset.sentiment;
        const newsItems = document.querySelectorAll('.news-item');
        
        newsItems.forEach(item => {
            if (sentiment === 'all') {
                item.style.display = 'block';
            } else if (sentiment === 'positive') {
                item.style.display = item.classList.contains('positive') ? 'block' : 'none';
            } else if (sentiment === 'negative') {
                item.style.display = item.classList.contains('negative') ? 'block' : 'none';
            }
        });
    });
});

// Load available stocks when page loads
document.addEventListener('DOMContentLoaded', loadStocks);

// Show stocks modal
document.getElementById('show-stocks-btn').addEventListener('click', () => {
    document.getElementById('stocks-modal').classList.add('show');
});

// Close modal
document.querySelector('.close').addEventListener('click', () => {
    document.getElementById('stocks-modal').classList.remove('show');
});

// Search stocks
document.getElementById('stock-search').addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase();
    const stockItems = document.querySelectorAll('.stock-item');
    
    stockItems.forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(searchTerm) ? 'grid' : 'none';
    });
});

// Load stocks function
async function loadStocks() {
    try {
        const response = await fetch('/stocks');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Populate datalist
        const datalist = document.getElementById('stock-list');
        datalist.innerHTML = data.stocks.map(stock => 
            `<option value="${stock.symbol}">${stock.name} (${stock.symbol})</option>`
        ).join('');
        
        // Populate stocks list
        const stocksList = document.getElementById('stocks-list');
        stocksList.innerHTML = data.stocks.map(stock => `
            <div class="stock-item" data-symbol="${stock.symbol}">
                <span>${stock.symbol}</span>
                <span>${stock.name}</span>
                <span>${stock.sector}</span>
                <span>$${stock.current_price.toFixed(2)}</span>
            </div>
        `).join('');
        
        // Add click handlers for stock items
        document.querySelectorAll('.stock-item').forEach(item => {
            item.addEventListener('click', () => {
                document.getElementById('symbol').value = item.dataset.symbol;
                document.getElementById('stocks-modal').classList.remove('show');
                document.getElementById('prediction-form').dispatchEvent(new Event('submit'));
            });
        });
        
    } catch (error) {
        console.error('Error loading stocks:', error);
    }
}

// Update the updateResults function
function updateResults(data) {
    document.getElementById('stock-symbol').textContent = data.symbol;
    document.getElementById('current-price').textContent = `$${data.current_price}`;
    document.getElementById('predicted-price').textContent = `$${data.prediction}`;
    
    // Enhanced prediction range display
    const priceChange = ((data.prediction - data.current_price) / data.current_price * 100).toFixed(2);
    const rangeSpread = ((data.upper_bound - data.lower_bound) / data.prediction * 100).toFixed(2);
    
    document.getElementById('prediction-range').innerHTML = `
        <div class="range-details">
            <span>Range: $${data.lower_bound} - $${data.upper_bound}</span>
            <span class="spread">Spread: ${rangeSpread}%</span>
        </div>
        <div class="change-prediction ${priceChange >= 0 ? 'positive' : 'negative'}">
            Expected Change: ${priceChange}%
        </div>
    `;
    
    // Update confidence indicators with momentum
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceLevel = document.getElementById('confidence-level');
    const trendDirection = document.getElementById('trend-direction');
    const trendStrength = document.getElementById('trend-strength');
    
    confidenceBar.style.width = `${data.confidence.level}%`;
    confidenceBar.className = `confidence-bar ${data.confidence.level > 70 ? 'high' : data.confidence.level > 40 ? 'medium' : 'low'}`;
    
    confidenceLevel.textContent = `${data.confidence.level.toFixed(1)}% Confidence`;
    
    const trend = data.confidence.trend;
    const trendIcon = trend.direction === 'up' ? '↑' : '↓';
    const trendClass = trend.direction === 'up' ? 'trend-up' : 'trend-down';
    
    trendDirection.className = trendClass;
    trendDirection.innerHTML = `
        ${trendIcon} ${Math.abs(trend.percentage).toFixed(2)}%
        <span class="momentum ${trend.momentum}">${trend.momentum}</span>
    `;
    
    trendStrength.textContent = `${trend.strength > 5 ? 'Strong' : 'Moderate'} ${trend.direction.toUpperCase()}`;
    
    // Update metrics with enhanced formatting
    document.getElementById('mae').innerHTML = `$${data.metrics.mae}<br><span class="metric-label">Mean Absolute Error</span>`;
    document.getElementById('rmse').innerHTML = `$${data.metrics.rmse}<br><span class="metric-label">Root Mean Square Error</span>`;
    document.getElementById('r2').innerHTML = `${data.metrics.r2}%<br><span class="metric-label">R² Score</span>`;
} 