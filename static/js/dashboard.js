/* ==========================================================================
   OptiCrop – Dashboard JavaScript
   Renders all interactive Chart.js visualizations for the Research Dashboard
   ========================================================================== */
(function () {
    "use strict";

    const data = window.DASHBOARD_DATA;
    if (!data) return;

    // Chart.js global defaults
    Chart.defaults.font.family = "'Plus Jakarta Sans', system-ui, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = getComputedStyle(document.documentElement)
        .getPropertyValue("--text-secondary").trim() || "#4b5563";

    // Color palette
    const GREEN = "#16a34a";
    const BLUE = "#2563eb";
    const ORANGE = "#f97316";
    const PURPLE = "#a855f7";
    const palette = [
        "#16a34a", "#2563eb", "#f97316", "#a855f7", "#06b6d4",
        "#eab308", "#ec4899", "#8b5cf6", "#14b8a6", "#f43f5e",
        "#84cc16", "#3b82f6", "#fb923c", "#c084fc", "#22d3ee",
        "#facc15", "#f472b6", "#a78bfa", "#2dd4bf", "#fb7185",
        "#a3e635", "#60a5fa"
    ];

    /* ----- 1. Crop Distribution Chart ----- */
    const cropLabels = data.crop_counts.map(function (c) { return c.crop; });
    const cropCounts = data.crop_counts.map(function (c) { return c.count; });

    new Chart(document.getElementById("cropDistChart"), {
        type: "bar",
        data: {
            labels: cropLabels,
            datasets: [{
                label: "Sample Count",
                data: cropCounts,
                backgroundColor: palette,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(20,83,45,0.9)",
                    padding: 12,
                    cornerRadius: 8,
                }
            },
            scales: {
                x: { ticks: { maxRotation: 60, minRotation: 45, font: { size: 10 } } },
                y: { beginAtZero: true, ticks: { precision: 0 } }
            }
        }
    });

    /* ----- 2. NPK Analysis per Crop ----- */
    const npkLabels = data.npk_by_crop.map(function (c) { return c.label; });
    const nData = data.npk_by_crop.map(function (c) { return c.N; });
    const pData = data.npk_by_crop.map(function (c) { return c.P; });
    const kData = data.npk_by_crop.map(function (c) { return c.K; });

    new Chart(document.getElementById("npkChart"), {
        type: "bar",
        data: {
            labels: npkLabels,
            datasets: [
                { label: "Nitrogen (N)", data: nData, backgroundColor: GREEN, borderRadius: 4 },
                { label: "Phosphorous (P)", data: pData, backgroundColor: BLUE, borderRadius: 4 },
                { label: "Potassium (K)", data: kData, backgroundColor: ORANGE, borderRadius: 4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "top", labels: { usePointStyle: true, padding: 16 } },
                tooltip: { backgroundColor: "rgba(20,83,45,0.9)", padding: 12, cornerRadius: 8 }
            },
            scales: {
                x: { ticks: { maxRotation: 60, minRotation: 45, font: { size: 10 } } },
                y: { beginAtZero: true }
            }
        }
    });

    /* ----- 3. Distribution Charts (Histograms) ----- */
    function makeHistogram(values, bins) {
        const min = Math.min.apply(null, values);
        const max = Math.max.apply(null, values);
        const step = (max - min) / bins;
        const labels = [];
        const counts = new Array(bins).fill(0);
        for (let i = 0; i < bins; i++) {
            labels.push((min + i * step).toFixed(1));
        }
        values.forEach(function (v) {
            let idx = Math.floor((v - min) / step);
            if (idx >= bins) idx = bins - 1;
            if (idx < 0) idx = 0;
            counts[idx]++;
        });
        return { labels: labels, counts: counts };
    }

    function distChart(canvasId, values, color, label) {
        const hist = makeHistogram(values, 20);
        new Chart(document.getElementById(canvasId), {
            type: "bar",
            data: {
                labels: hist.labels,
                datasets: [{
                    label: label,
                    data: hist.counts,
                    backgroundColor: color,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { backgroundColor: "rgba(20,83,45,0.9)", padding: 10, cornerRadius: 8 }
                },
                scales: {
                    x: { ticks: { font: { size: 9 }, maxTicksLimit: 10 } },
                    y: { beginAtZero: true, ticks: { precision: 0 } }
                }
            }
        });
    }

    distChart("rainfallChart", data.distributions.rainfall, BLUE, "Rainfall (mm)");
    distChart("tempChart", data.distributions.temperature, ORANGE, "Temperature (°C)");
    distChart("humidityChart", data.distributions.humidity, PURPLE, "Humidity (%)");
    distChart("phChart", data.distributions.ph, GREEN, "pH");

    /* ----- 4. Correlation Heatmap ----- */
    const corr = data.correlation;
    const wrapper = document.getElementById("heatmapWrapper");
    if (wrapper && corr) {
        const cols = corr.columns;
        let html = '<table class="heatmap-table"><thead><tr><th></th>';
        cols.forEach(function (c) {
            html += '<th>' + c.charAt(0).toUpperCase() + '</th>';
        });
        html += '</tr></thead><tbody>';
        for (let i = 0; i < cols.length; i++) {
            html += '<tr><th>' + cols[i].charAt(0).toUpperCase() + cols[i].slice(1) + '</th>';
            for (let j = 0; j < cols.length; j++) {
                const val = corr.values[i][j];
                const intensity = Math.abs(val);
                // Green for positive, orange for negative
                let bg;
                if (val >= 0) {
                    bg = 'rgba(34,197,94,' + (0.1 + intensity * 0.7) + ')';
                } else {
                    bg = 'rgba(249,115,22,' + (0.1 + intensity * 0.7) + ')';
                }
                const textColor = intensity > 0.5 ? '#fff' : 'var(--text-primary)';
                html += '<td style="background:' + bg + '; color:' + textColor + '">' + val.toFixed(2) + '</td>';
            }
            html += '</tr>';
        }
        html += '</tbody></table>';
        wrapper.innerHTML = html;
    }
})();
