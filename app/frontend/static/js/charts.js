/**
 * 📊 Charts Module for VoxDynamics — Premium Edition
 * 5 interactive charts with consistent dark theme styling.
 */

const EMOTION_COLORS = {
    happy: '#00ff87', sad: '#667eea', angry: '#ff416c',
    fearful: '#f9a825', surprised: '#ab47bc', neutral: '#9aa0b8',
    disgust: '#ff8c00', calm: '#00e5ff', silence: '#2d2d38'
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
function renderTimelineChart(containerId, allSegments) {
    if (!allSegments.length) return;

    // Use ALL segments (speech + silence) to ensure correct time gaps
    const times = allSegments.map(s => s.time_s);

    // Extract emotion labels from the first speech segment found
    const firstSpeech = allSegments.find(s => s.is_speech);
    if (!firstSpeech || !firstSpeech.scores) return;
    const emotions = Object.keys(firstSpeech.scores);

    const traces = emotions.map((emo) => {
        const color = EMOTION_COLORS[emo] || '#9aa0b8';
        // Map scores: if silence, score is 0
        const vals = allSegments.map(s => (s.is_speech && s.scores) ? (s.scores[emo] || 0) : 0);
        const rgb = hexToRgb(color);

        return {
            type: 'scatter', mode: 'lines', name: emo.charAt(0).toUpperCase() + emo.slice(1),
            x: times, y: vals, stackgroup: 'one',
            line: { color, width: 2, shape: 'spline' },
            fillcolor: `rgba(${rgb.r},${rgb.g},${rgb.b},0.35)`,
            hoverinfo: 'text',
            hovertext: allSegments.map((s) => {
                if (!s.is_speech) return `<b>SILENCE</b><br>⏱ ${s.time_s.toFixed(2)}s`;
                return `<b>${emo.toUpperCase()}</b><br>⏱ ${s.time_s.toFixed(2)}s<br>🎯 Prob: <b>${(s.scores[emo] * 100).toFixed(1)}%</b>`;
            }),
        };
    });

    const layout = {
        paper_bgcolor: BG_T,
        plot_bgcolor: 'rgba(10,10,25,0.4)',
        margin: { t: 10, b: 36, l: 38, r: 10 },
        height: 260,
        xaxis: {
            title: { text: 'Time (s)', font: { color: TICK, size: 11 } },
            gridcolor: GRID,
            tickfont: { color: TICK },
            zeroline: false
        },
        yaxis: {
            range: [0, 1.05],
            gridcolor: GRID,
            tickfont: { color: TICK },
            zeroline: false,
            title: { text: 'Confidence', font: { color: TICK, size: 10 } }
        },
        legend: {
            orientation: 'h',
            y: 1.15,
            x: 0.5,
            xanchor: 'center',
            font: { color: '#9aa0b8', size: 10 },
            bgcolor: BG_T
        },
        font: PFONT,
        hovermode: 'x unified',
        hoverlabel: { bgcolor: '#1a1a2e', font: { family: 'Outfit', color: '#fff' } },
    };

    Plotly.newPlot(containerId, traces, layout, PCFG);
}

/* ═══════════════════════════════════════════════════════
   3. EMOTION+CONFIDENCE COMBO — proportional, spread waveform
      Each segment occupies equal visual space based on duration ratio
═══════════════════════════════════════════════════════ */
function renderEmotionConfidenceChart(containerId, allSegs) {
    const totalDuration = allSegs.reduce((acc, s) => acc + (s.duration_s || 0.5), 0);
    if (totalDuration === 0) return;

    const RESOLUTION = 600; // total data points across chart

    // Build peaks array: for each of the 600 points, find the matching segment
    const peaks = new Float32Array(RESOLUTION);
    const segForPoint = new Array(RESOLUTION);

    // Precompute each segment's normalized start/end position (0-1)
    let cursor = 0;
    const segLayout = allSegs.map(s => {
        const dur = s.duration_s || 0.5;
        const normStart = cursor / totalDuration;
        const normEnd = (cursor + dur) / totalDuration;
        cursor += dur;
        return { ...s, normStart, normEnd };
    });

    function seededRand(seed) {
        let x = Math.sin(seed * 9301 + 49297) * 49297;
        return x - Math.floor(x);
    }

    for (let i = 0; i < RESOLUTION; i++) {
        const pos = i / RESOLUTION; // normalized position 0-1
        const seg = segLayout.find(s => pos >= s.normStart && pos < s.normEnd);
        segForPoint[i] = seg || null;

        if (!seg || !seg.is_speech) {
            // Silence: very tiny noise floor
            peaks[i] = seededRand(i * 7) * 0.02;
        } else {
            const progress = (pos - seg.normStart) / (seg.normEnd - seg.normStart);
            const si = allSegs.indexOf(seg);
            const seed = si * 1000 + i;
            const burst = seededRand(seed) * 0.45;
            const wave = Math.sin(progress * Math.PI * 10) * 0.25 + 0.28;
            const envelope = Math.pow(Math.sin(progress * Math.PI), 0.6) * 0.9;
            peaks[i] = Math.max(0.03, (burst + wave) * envelope * (seg.confidence || 0.8));
        }
    }

    renderWaveformEmotionChart(containerId, peaks, totalDuration, allSegs, segLayout);
}

/* ═══════════════════════════════════════════════════════
   4. RADAR CHART — emotional dimension profile
═══════════════════════════════════════════════════════ */
function renderRadarChart(containerId, summary) {
    if (!summary.avg_scores) return;

    const dataArr = Object.entries(summary.avg_scores);
    // Sort logically for radar shape consistency (e.g. alphabetical)
    dataArr.sort((a, b) => a[0].localeCompare(b[0]));

    const labels = dataArr.map(([l]) => l.charAt(0).toUpperCase() + l.slice(1));
    const values = dataArr.map(([, v]) => v);

    // Connect back to start for closed loop
    labels.push(labels[0]);
    values.push(values[0]);

    const domColor = EMOTION_COLORS[summary.dominant_emotion] || '#00e5ff';
    const rgb = hexToRgb(domColor);

    Plotly.newPlot(containerId, [{
        type: 'scatterpolar',
        r: values,
        theta: labels,
        fill: 'toself',
        fillcolor: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.2)`,
        line: { color: domColor, width: 3 },
        marker: { color: domColor, size: 7 },
        hoverinfo: 'r+theta',
    }], {
        paper_bgcolor: BG_T, plot_bgcolor: BG_T,
        margin: { t: 30, b: 30, l: 50, r: 50 }, height: 260, showlegend: false,
        polar: {
            bgcolor: 'rgba(10,10,25,0.4)',
            radialaxis: {
                visible: true,
                range: [0, Math.max(...values) + 0.15],
                tickfont: { color: TICK, size: 9 },
                gridcolor: 'rgba(255,255,255,0.06)',
                linecolor: 'rgba(255,255,255,0.06)'
            },
            angularaxis: {
                tickfont: { color: '#ccc', size: 10, family: 'Outfit' },
                gridcolor: 'rgba(255,255,255,0.06)',
                linecolor: 'rgba(255,255,255,0.06)'
            }
        },
        font: PFONT,
    }, PCFG);
}

/* ═══════════════════════════════════════════════════════
   5. WAVEFORM EMOTION CHART — proportional per-segment coloring
      X axis = normalized position (0→1), fills chart fully
═══════════════════════════════════════════════════════ */
function renderWaveformEmotionChart(containerId, peaks, totalDuration, allSegs, segLayout) {
    const N = peaks.length;

    // If segLayout not provided (backward compat), build one from real timestamps
    if (!segLayout) {
        let cur = 0;
        segLayout = allSegs.map(s => {
            const dur = s.duration_s || 0.5;
            const ns = cur / totalDuration, ne = (cur + dur) / totalDuration;
            cur += dur;
            return { ...s, normStart: ns, normEnd: ne };
        });
    }

    // Normalize peaks to 0-1
    const maxP = Math.max(...peaks) || 1;
    const normPeaks = Array.from(peaks).map(p => p / maxP);

    // X axis = normalized position 0→1
    const xs = normPeaks.map((_, i) => i / N);

    // ── Build colour-coded trace segments ───────────────
    const traces = [];
    let curSeg = null, curKey = null;
    let bx = [], byPos = [], byNeg = [], bHover = [];

    function flushSeg() {
        if (!bx.length) return;
        let col, isSilence;
        if (!curSeg || !curSeg.is_speech) {
            col = '#2a2a3d';
            isSilence = true;
        } else {
            col = EMOTION_COLORS[curSeg.emotion_label] || '#00e5ff';
            isSilence = false;
        }
        const rgb = hexToRgb(col.startsWith('#') ? col : '#555');

        // Positive fill
        traces.push({
            type: 'scatter', mode: 'lines', x: [...bx], y: [...byPos],
            fill: 'tozeroy',
            fillcolor: isSilence ? 'rgba(30,30,50,0.3)' : `rgba(${rgb.r},${rgb.g},${rgb.b},0.3)`,
            line: { color: isSilence ? 'rgba(80,80,100,0.25)' : col, width: isSilence ? 0.5 : 1.5, shape: 'spline', smoothing: 0.5 },
            hovertext: [...bHover], hoverinfo: 'text',
            hoverlabel: { bgcolor: '#0d0d20', bordercolor: col, font: { family: 'Outfit', color: '#fff', size: 12 } },
            showlegend: false,
        });
        // Mirror (negative)
        traces.push({
            type: 'scatter', mode: 'lines', x: [...bx], y: [...byNeg],
            fill: 'tozeroy',
            fillcolor: isSilence ? 'rgba(20,20,40,0.2)' : `rgba(${rgb.r},${rgb.g},${rgb.b},0.15)`,
            line: { color: isSilence ? 'rgba(60,60,80,0.15)' : `rgba(${rgb.r},${rgb.g},${rgb.b},0.5)`, width: isSilence ? 0.3 : 0.8, shape: 'spline', smoothing: 0.5 },
            hoverinfo: 'skip', showlegend: false,
        });
    }

    for (let i = 0; i < N; i++) {
        const pos = xs[i];
        const seg = segLayout.find(s => pos >= s.normStart && pos < s.normEnd);
        const key = seg ? (seg.emotion_label + seg.normStart) : '__silence__';

        if (key !== curKey) {
            flushSeg();
            curSeg = seg; curKey = key;
            bx = []; byPos = []; byNeg = []; bHover = [];
        }

        bx.push(pos);
        byPos.push(normPeaks[i]);
        byNeg.push(-normPeaks[i] * 0.7);

        if (seg && seg.is_speech) {
            const realTime = seg.time_s + (pos - seg.normStart) * totalDuration;
            bHover.push(
                `<b>${seg.emoji || ''} ${(seg.emotion_label || '').charAt(0).toUpperCase() + (seg.emotion_label || '').slice(1)}</b><br>` +
                `⏱ ${realTime.toFixed(2)}s · dur: ${(seg.duration_s || 0).toFixed(2)}s<br>` +
                `🎯 Confidence: <b>${((seg.confidence || 0) * 100).toFixed(1)}%</b><br>` +
                `A: ${(seg.arousal || 0.5).toFixed(2)} · D: ${(seg.dominance || 0.5).toFixed(2)} · V: ${(seg.valence || 0.5).toFixed(2)}`
            );
        } else {
            bHover.push(seg ? `<i>Silence</i> (${seg.time_s.toFixed(1)}s – ${(seg.time_s + (seg.duration_s || 0)).toFixed(1)}s)` : '');
        }
    }
    flushSeg();

    // ── Background rects & separator lines ──────────────
    const shapes = [];
    segLayout.forEach(s => {
        const col = s.is_speech ? (EMOTION_COLORS[s.emotion_label] || '#555') : '#22223a';
        const rgb = hexToRgb(col.startsWith('#') ? col : '#555');

        // Background fill
        shapes.push({
            type: 'rect',
            x0: s.normStart, x1: s.normEnd, y0: -0.85, y1: 1.0,
            fillcolor: s.is_speech ? `rgba(${rgb.r},${rgb.g},${rgb.b},0.07)` : 'rgba(10,10,20,0.15)',
            line: { width: 0 },
        });
        // Left edge line
        shapes.push({
            type: 'line',
            x0: s.normStart, x1: s.normStart, y0: -0.85, y1: 1.0,
            line: { color: s.is_speech ? `rgba(${rgb.r},${rgb.g},${rgb.b},0.4)` : 'rgba(50,50,70,0.3)', width: s.is_speech ? 1.5 : 0.5, dash: s.is_speech ? 'solid' : 'dot' },
        });
    });

    // ── Annotations: emotion label + confidence ──────────
    const speechSegs = segLayout.filter(s => s.is_speech);
    const annotations = speechSegs.map(s => {
        const col = EMOTION_COLORS[s.emotion_label] || '#ccc';
        const midX = (s.normStart + s.normEnd) / 2;
        const label = (s.emotion_label || '').charAt(0).toUpperCase() + (s.emotion_label || '').slice(1);
        return {
            x: midX, y: 1.05, yref: 'paper', showarrow: false, xanchor: 'center',
            text: `<b>${label}</b><br><span style="font-size:9px;opacity:0.9">${((s.confidence || 0) * 100).toFixed(0)}%</span>`,
            font: { family: 'Outfit', size: 11, color: col },
        };
    });

    // ── Custom tick labels: real time at each segment boundary ──
    const tickVals = [...new Set(segLayout.map(s => s.normStart))];
    const tickText = tickVals.map(v => {
        const seg = segLayout.find(s => s.normStart === v);
        return seg ? `${seg.time_s.toFixed(1)}s` : '';
    });
    // Add final end
    tickVals.push(1.0);
    const lastSeg = segLayout[segLayout.length - 1];
    tickText.push(lastSeg ? `${(lastSeg.time_s + (lastSeg.duration_s || 0)).toFixed(1)}s` : '');

    Plotly.newPlot(containerId, traces, {
        paper_bgcolor: BG_T,
        plot_bgcolor: 'rgba(6,6,18,0.6)',
        margin: { t: 52, b: 42, l: 10, r: 10 },
        height: 240,
        xaxis: {
            title: { text: 'Audio Timeline (proportional)', font: { color: TICK, size: 10 } },
            range: [0, 1],
            tickvals: tickVals,
            ticktext: tickText,
            tickfont: { color: TICK, size: 9 },
            gridcolor: 'rgba(255,255,255,0.025)',
            zeroline: false,
        },
        yaxis: {
            showticklabels: false,
            zeroline: true, zerolinecolor: 'rgba(255,255,255,0.12)', zerolinewidth: 1,
            gridcolor: 'rgba(255,255,255,0.02)',
            range: [-0.9, 1.2],
            fixedrange: true,
        },
        shapes, annotations,
        showlegend: false,
        font: PFONT,
        hovermode: 'x unified',
        hoverlabel: { bgcolor: '#0d0d20', font: { family: 'Outfit', color: '#fff', size: 12 }, align: 'left' },
    }, PCFG);
}



/* ═══════════════════════════════════════════════════════
   UTILITY
═══════════════════════════════════════════════════════ */
function hexToRgb(hex) {
    return { r: parseInt(hex.slice(1, 3), 16), g: parseInt(hex.slice(3, 5), 16), b: parseInt(hex.slice(5, 7), 16) };
}
