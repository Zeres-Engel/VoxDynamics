/**
 * 📊 Charts Module for VoxDynamics — Premium Edition
 * 5 interactive charts with consistent dark theme styling.
 */

const EMOTION_COLORS = {
    happy: '#00ff87', sad: '#667eea', angry: '#ff416c',
    fearful: '#f9a825', surprised: '#ab47bc', neutral: '#9aa0b8',
    disgust: '#ff8c00', calm: '#00e5ff', fear: '#f9a825', surprise: '#ab47bc',
};

const PCFG = { displayModeBar: false, responsive: true };
const BG_T = 'transparent';
const PFONT = { family: 'Outfit', color: '#9aa0b8' };
const GRID = 'rgba(255,255,255,0.04)';
const TICK = '#565d7a';

/* ═══════════════════════════════════════════════════════
   1. EMOTION DONUT
═══════════════════════════════════════════════════════ */
function renderPieChart(containerId, speechSegs) {
    const counts = {};
    speechSegs.forEach(s => { counts[s.emotion_label] = (counts[s.emotion_label] || 0) + 1; });

    const labels = Object.keys(counts);
    const values = Object.values(counts);
    const colors = labels.map(l => EMOTION_COLORS[l] || '#555');

    Plotly.newPlot(containerId, [{
        type: 'pie', labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
        values, hole: 0.65,
        marker: { colors, line: { color: '#0d0d1e', width: 3 } },
        textinfo: 'percent', textposition: 'inside',
        textfont: { family: 'Outfit', color: '#fff', size: 13 },
        hoverinfo: 'label+value+percent',
        hoverlabel: { bgcolor: '#1a1a2e', bordercolor: '#333', font: { family: 'Outfit', color: '#fff' } },
        sort: false,
    }], {
        paper_bgcolor: BG_T, plot_bgcolor: BG_T,
        margin: { t: 10, b: 45, l: 10, r: 10 }, height: 260,
        showlegend: true,
        legend: { orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center', font: { family: 'Outfit', color: '#9aa0b8', size: 11 } },
        font: PFONT,
    }, PCFG);
}

/* ═══════════════════════════════════════════════════════
   2. SENTIMENT WAVE — dimension timeline
═══════════════════════════════════════════════════════ */
function renderTimelineChart(containerId, speechSegs) {
    const times = speechSegs.map(s => s.time_s);
    const dims = [
        { key: 'arousal', color: '#f9a825', name: 'Arousal' },
        { key: 'dominance', color: '#ab47bc', name: 'Dominance' },
        { key: 'valence', color: '#00e5ff', name: 'Valence' },
    ];

    const traces = dims.map(d => {
        const rgb = hexToRgb(d.color);
        return {
            type: 'scatter', mode: 'lines', x: times,
            y: speechSegs.map(s => s[d.key]), name: d.name,
            line: { color: d.color, width: 2.5, shape: 'spline' },
            fill: 'tozeroy', fillcolor: `rgba(${rgb.r},${rgb.g},${rgb.b},0.06)`,
        };
    });

    Plotly.newPlot(containerId, traces, {
        paper_bgcolor: BG_T, plot_bgcolor: 'rgba(10,10,25,0.4)',
        margin: { t: 10, b: 36, l: 38, r: 10 }, height: 260,
        xaxis: { title: { text: 'Time (s)', font: { color: TICK, size: 11 } }, gridcolor: GRID, tickfont: { color: TICK }, zeroline: false },
        yaxis: { range: [0, 1.05], gridcolor: GRID, tickfont: { color: TICK }, zeroline: false },
        legend: { orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center', font: { color: '#9aa0b8', size: 11 }, bgcolor: BG_T },
        font: PFONT,
    }, PCFG);
}

/* ═══════════════════════════════════════════════════════
   3. EMOTION+CONFIDENCE COMBO — interactive bar segments
      Replaces both "Emotion Segment Map" and "Confidence Per Segment"
═══════════════════════════════════════════════════════ */
function renderEmotionConfidenceChart(containerId, speechSegs) {
    // Generate synthetic waveform peaks that look like real audio, then
    // delegate to renderWaveformEmotionChart for identical visual output.
    const POINTS_PER_SEG = 80; // enough points for smooth curves
    const totalDuration = speechSegs.length > 0
        ? speechSegs[speechSegs.length - 1].time_s + 0.5
        : 2;

    const fakePeaks = [];

    // Seeded pseudo-random for deterministic results per segment
    function seededRand(seed) {
        let x = Math.sin(seed * 9301 + 49297) * 49297;
        return x - Math.floor(x);
    }

    speechSegs.forEach((s, si) => {
        const conf = s.confidence;
        for (let i = 0; i < POINTS_PER_SEG; i++) {
            const t = i / POINTS_PER_SEG;
            // Combine multiple frequencies to simulate real speech energy
            const seed = si * 1000 + i;
            const burst = seededRand(seed) * 0.5;
            const wave1 = Math.sin(t * Math.PI * 6 + si * 2.1) * 0.25;
            const wave2 = Math.sin(t * Math.PI * 14 + si * 0.7) * 0.12;
            const wave3 = Math.cos(t * Math.PI * 3 + si * 1.4) * 0.15;
            // Envelope: fade in/out at segment edges for natural look
            const envelope = Math.sin(t * Math.PI) * 0.6 + 0.4;
            const peak = Math.abs((burst + wave1 + wave2 + wave3) * envelope * conf);
            fakePeaks.push(Math.min(1, peak));
        }
    });

    renderWaveformEmotionChart(containerId, fakePeaks, totalDuration, speechSegs);
}

/* ═══════════════════════════════════════════════════════
   4. RADAR CHART — emotional dimension profile
═══════════════════════════════════════════════════════ */
function renderRadarChart(containerId, summary) {
    const dims = ['Arousal', 'Dominance', 'Valence', 'Arousal'];
    const vals = [summary.avg_arousal, summary.avg_dominance, summary.avg_valence, summary.avg_arousal];

    Plotly.newPlot(containerId, [{
        type: 'scatterpolar', r: vals, theta: dims,
        fill: 'toself', fillcolor: 'rgba(0, 229, 255, 0.12)',
        line: { color: '#00e5ff', width: 2.5 },
        marker: { color: '#00ff87', size: 8 },
        hoverinfo: 'r+theta',
    }], {
        paper_bgcolor: BG_T, plot_bgcolor: BG_T,
        margin: { t: 30, b: 30, l: 50, r: 50 }, height: 260, showlegend: false,
        polar: {
            bgcolor: 'rgba(10,10,25,0.4)',
            radialaxis: { visible: true, range: [0, 1], tickfont: { color: TICK, size: 10 }, gridcolor: 'rgba(255,255,255,0.06)', linecolor: 'rgba(255,255,255,0.06)' },
            angularaxis: { tickfont: { color: '#ccc', size: 12, family: 'Outfit' }, gridcolor: 'rgba(255,255,255,0.06)', linecolor: 'rgba(255,255,255,0.06)' }
        },
        font: PFONT,
    }, PCFG);
}

/* ═══════════════════════════════════════════════════════
   5. WAVEFORM EMOTION CHART — smooth, colored per-segment waveform
═══════════════════════════════════════════════════════ */
function renderWaveformEmotionChart(containerId, peaks, totalDuration, speechSegs) {
    const N = peaks.length;

    // Normalize peaks to 0-1 range so waveform fills chart
    const maxP = Math.max(...peaks) || 1;
    const normPeaks = peaks.map(p => p / maxP);

    // Map each peak to a time
    const times = normPeaks.map((_, i) => (i / N) * totalDuration);

    // Build segment lookup
    const segRanges = speechSegs.map(s => ({
        start: s.time_s, end: s.time_s + 0.5,
        label: s.emotion_label, confidence: s.confidence,
        arousal: s.arousal, dominance: s.dominance, valence: s.valence,
        emoji: s.emoji || '',
    }));

    // Group consecutive points by segment for per-segment coloring
    const traces = [];
    let currentSeg = null;
    let segTimes = [], segYpos = [], segYneg = [], segHover = [];

    function flushTrace() {
        if (!segTimes.length) return;
        const c = currentSeg ? (EMOTION_COLORS[currentSeg.label] || '#00e5ff') : 'rgba(100,100,120,0.3)';
        const rgb = hexToRgb(c.startsWith('#') ? c : '#666');

        // Positive half (main waveform)
        traces.push({
            type: 'scatter', mode: 'lines', x: segTimes, y: segYpos,
            fill: 'tozeroy', fillcolor: `rgba(${rgb.r},${rgb.g},${rgb.b},0.35)`,
            line: { color: c, width: 1 },
            hovertext: segHover, hoverinfo: 'text',
            hoverlabel: { bgcolor: '#12122a', bordercolor: c, font: { family: 'Outfit', color: '#fff', size: 12 } },
            showlegend: false,
        });
        // Negative half (mirror reflection, dimmer)
        traces.push({
            type: 'scatter', mode: 'lines', x: segTimes, y: segYneg,
            fill: 'tozeroy', fillcolor: `rgba(${rgb.r},${rgb.g},${rgb.b},0.15)`,
            line: { color: `rgba(${rgb.r},${rgb.g},${rgb.b},0.5)`, width: 1 },
            hoverinfo: 'skip', showlegend: false,
        });
    }

    for (let i = 0; i < N; i++) {
        const t = times[i];
        const seg = segRanges.find(s => t >= s.start && t < s.end);
        const segKey = seg ? seg.label + seg.start : '__none__';
        const prevKey = currentSeg ? currentSeg.label + currentSeg.start : '__none__';

        if (segKey !== prevKey) {
            // Flush previous segment trace
            flushTrace();
            currentSeg = seg;
            segTimes = []; segYpos = []; segYneg = []; segHover = [];
        }

        segTimes.push(t);
        segYpos.push(normPeaks[i]);
        segYneg.push(-normPeaks[i] * 0.7);

        if (seg) {
            segHover.push(
                `<b>${seg.emoji} ${seg.label.charAt(0).toUpperCase() + seg.label.slice(1)}</b><br>` +
                `⏱ ${t.toFixed(2)}s<br>` +
                `🎯 Confidence: <b>${(seg.confidence * 100).toFixed(1)}%</b><br>` +
                `A: ${seg.arousal.toFixed(3)} · D: ${seg.dominance.toFixed(3)} · V: ${seg.valence.toFixed(3)}`
            );
        } else {
            segHover.push(`${t.toFixed(2)}s — <i>silence</i>`);
        }
    }
    flushTrace(); // flush last group

    // Segment boundary lines
    const shapes = speechSegs.map(s => ({
        type: 'line', x0: s.time_s, x1: s.time_s, y0: -0.8, y1: 1.1,
        line: { color: 'rgba(255,255,255,0.15)', width: 1, dash: 'dot' },
    }));

    // Emotion label annotations on top
    const annotations = speechSegs.map(s => {
        const c = EMOTION_COLORS[s.emotion_label] || '#ccc';
        return {
            x: s.time_s + 0.25, y: 1.02, yref: 'paper', showarrow: false,
            text: `<b>${s.emotion_label.charAt(0).toUpperCase() + s.emotion_label.slice(1)}</b><br><span style="font-size:9px">${(s.confidence * 100).toFixed(0)}%</span>`,
            font: { family: 'Outfit', size: 10, color: c },
        };
    });

    Plotly.newPlot(containerId, traces, {
        paper_bgcolor: BG_T,
        plot_bgcolor: 'rgba(8,8,20,0.5)',
        margin: { t: 32, b: 36, l: 10, r: 10 },
        height: 200,
        xaxis: {
            title: { text: 'Time (s)', font: { color: TICK, size: 11 } },
            gridcolor: 'rgba(255,255,255,0.03)', tickfont: { color: TICK }, zeroline: false,
        },
        yaxis: {
            showticklabels: false, zeroline: false,
            gridcolor: 'rgba(255,255,255,0.02)',
            range: [-0.85, 1.15],
            fixedrange: true,
        },
        shapes,
        annotations,
        showlegend: false,
        font: PFONT,
        hovermode: 'closest',
    }, PCFG);
}

/* ═══════════════════════════════════════════════════════
   UTILITY
═══════════════════════════════════════════════════════ */
function hexToRgb(hex) {
    return { r: parseInt(hex.slice(1, 3), 16), g: parseInt(hex.slice(3, 5), 16), b: parseInt(hex.slice(5, 7), 16) };
}
