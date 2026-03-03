/**
 * ⚡ Main Application Logic — VoxDynamics Premium
 * Handles API calls, overlay control, WaveSurfer, data rendering, and auto-scroll.
 */

let selectedFile = null;
let overlayTimer = null;
let wavesurfer = null;
let audioPeaks = null;
let audioDuration = 0;

document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadHistory();
    setupDropzone();

    // Listen for chart seek events
    window.addEventListener('vox-seek', (e) => {
        const time = e.detail.time;
        if (wavesurfer) {
            const duration = wavesurfer.getDuration();
            if (duration > 0) {
                wavesurfer.seekTo(time / duration);
                if (!wavesurfer.isPlaying()) wavesurfer.play();
            }
        }
        highlightTableRow(time);
    });
});

function highlightTableRow(time) {
    const tbody = document.getElementById('segment-tbody');
    const rows = tbody.querySelectorAll('tr');
    let targetRow = null;

    // Find the closest row (segments are 0.5s)
    rows.forEach(row => {
        row.classList.remove('active-segment');
        const rowTime = parseFloat(row.cells[1].textContent);
        if (Math.abs(rowTime - time) < 0.26) {
            targetRow = row;
        }
    });

    if (targetRow) {
        targetRow.classList.add('active-segment');
        targetRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

/* ═══════════════════════════════════════════════════════
   HEALTH CHECK
═══════════════════════════════════════════════════════ */
async function checkHealth() {
    try {
        const r = await fetch('/health');
        const d = await r.json();
        const el = document.getElementById('model-status-text');
        const pill = document.getElementById('model-status');
        if (d.models_loaded) {
            el.textContent = 'AI Models Ready';
            pill.style.background = 'rgba(0,255,135,0.08)';
            pill.style.borderColor = 'rgba(0,255,135,0.25)';
            pill.style.color = '#00ff87';
        } else {
            el.textContent = 'Models Loading...';
            pill.style.borderColor = 'rgba(249,168,37,0.3)';
            pill.style.color = '#f9a825';
            pill.style.background = 'rgba(249,168,37,0.07)';
            setTimeout(checkHealth, 3000);
        }
    } catch {
        document.getElementById('model-status-text').textContent = 'Server Unreachable';
    }
}

/* ═══════════════════════════════════════════════════════
   DROPZONE & WAVEFORM
═══════════════════════════════════════════════════════ */
function setupDropzone() {
    const zone = document.getElementById('drop-zone');
    const input = document.getElementById('audio-file-input');

    input.addEventListener('change', e => handleFile(e.target.files[0]));
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
}

function handleFile(file) {
    if (!file) return;
    selectedFile = file;

    document.getElementById('file-name-text').textContent =
        `${file.name}  (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    document.getElementById('audio-preview-wrap').style.display = 'block';

    if (!wavesurfer) {
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: 'rgba(0, 229, 255, 0.35)',
            progressColor: '#00e5ff',
            cursorColor: '#00ff87',
            barWidth: 2,
            barGap: 3,
            barRadius: 2,
            height: 50,
            normalize: true,
        });

        const playBtn = document.getElementById('play-pause-btn');
        wavesurfer.on('play', () => playBtn.innerHTML = '<i class="ph-fill ph-pause"></i>');
        wavesurfer.on('pause', () => playBtn.innerHTML = '<i class="ph-fill ph-play"></i>');
        wavesurfer.on('audioprocess', () => {
            document.getElementById('time-display').textContent =
                formatTime(wavesurfer.getCurrentTime()) + ' / ' + formatTime(wavesurfer.getDuration());
        });
        wavesurfer.on('ready', () => {
            document.getElementById('time-display').textContent =
                '0:00 / ' + formatTime(wavesurfer.getDuration());
        });
        playBtn.addEventListener('click', () => wavesurfer.playPause());
    }

    const blobUrl = URL.createObjectURL(file);
    wavesurfer.load(blobUrl);

    // Extract waveform peaks for the emotion waveform chart
    const reader = new FileReader();
    reader.onload = async (evt) => {
        try {
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const buffer = await audioCtx.decodeAudioData(evt.target.result);
            const raw = buffer.getChannelData(0);
            audioDuration = buffer.duration;

            // Downsample to ~600 points (envelope)
            const N = 600;
            const block = Math.floor(raw.length / N);
            audioPeaks = [];
            for (let i = 0; i < N; i++) {
                let peak = 0;
                for (let j = 0; j < block; j++) {
                    const v = Math.abs(raw[i * block + j]);
                    if (v > peak) peak = v;
                }
                audioPeaks.push(peak);
            }
            audioCtx.close();
        } catch (e) {
            console.warn('Could not extract peaks:', e);
            audioPeaks = null;
        }
    };
    reader.readAsArrayBuffer(file);

    document.getElementById('analyze-btn').disabled = false;
    document.getElementById('results-section').style.display = 'none';
    setStatus('File ready. Press "Start Deep Analysis" to begin.', 'info');
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
}

/* ═══════════════════════════════════════════════════════
   OVERLAY — Fullscreen Loading Modal
═══════════════════════════════════════════════════════ */
function showOverlay() {
    const overlay = document.getElementById('analysis-overlay');
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';

    ['step-upload', 'step-vad', 'step-emotion', 'step-report'].forEach(id => {
        document.getElementById(id).classList.remove('active', 'done');
    });

    const bar = document.getElementById('overlay-bar');
    bar.style.transition = 'none';
    bar.style.width = '0%';

    setOverlayStep('step-upload', 0);
    setOverlayStep('step-vad', 1200);
    setOverlayStep('step-emotion', 3500);
    setOverlayStep('step-report', 7000);
}

function setOverlayStep(stepId, delay) {
    const steps = ['step-upload', 'step-vad', 'step-emotion', 'step-report'];
    const idx = steps.indexOf(stepId);

    overlayTimer = setTimeout(() => {
        if (idx > 0) {
            document.getElementById(steps[idx - 1]).classList.remove('active');
            document.getElementById(steps[idx - 1]).classList.add('done');
        }
        document.getElementById(stepId).classList.add('active');

        const pct = Math.round(((idx + 1) / steps.length) * 90);
        const bar = document.getElementById('overlay-bar');
        bar.style.transition = 'width 1.5s cubic-bezier(0.25,1,0.5,1)';
        bar.style.width = `${pct}%`;
    }, delay);
}

function hideOverlay() {
    clearTimeout(overlayTimer);
    const bar = document.getElementById('overlay-bar');
    bar.style.transition = 'width 0.4s ease';
    bar.style.width = '100%';

    ['step-upload', 'step-vad', 'step-emotion', 'step-report'].forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active');
        el.classList.add('done');
    });

    setTimeout(() => {
        const overlay = document.getElementById('analysis-overlay');
        overlay.style.animation = 'overlayOut 0.4s ease forwards';
        setTimeout(() => {
            overlay.classList.remove('active');
            overlay.style.animation = '';
            document.body.style.overflow = '';
        }, 400);
    }, 500);
}

/* ═══════════════════════════════════════════════════════
   ANALYSIS
═══════════════════════════════════════════════════════ */
async function startAnalysis() {
    if (!selectedFile) return;

    setStatus('', 'info');
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('analyze-btn').disabled = true;
    showOverlay();

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const res = await fetch('/api/analyze', { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(err.detail || 'Analysis pipeline failed');
        }
        const data = await res.json();

        hideOverlay();
        setTimeout(() => renderResults(data), 600);

    } catch (e) {
        hideOverlay();
        setTimeout(() => {
            setStatus(`❌ ${e.message}`, 'error');
            document.getElementById('analyze-btn').disabled = false;
        }, 500);
    }
}

/* ═══════════════════════════════════════════════════════
   RENDER RESULTS
═══════════════════════════════════════════════════════ */
function renderResults(data) {
    const { segments, summary } = data;
    const speechSegs = segments.filter(s => s.is_speech);

    if (!speechSegs.length) {
        setStatus('❌ No speech detected in this audio file.', 'error');
        document.getElementById('analyze-btn').disabled = false;
        return;
    }

    // Re-index time_s sequentially to ensure consistent chart rendering
    speechSegs.forEach((s, i) => { s.time_s = i * 0.5; });

    // Dominant Emotion
    document.getElementById('dominant-emotion').textContent =
        `${summary.dominant_emoji} ${summary.dominant_emotion.toUpperCase()}`;

    // Dominant stats — show all emotion counts as mini badges
    const emotionCounts = {};
    speechSegs.forEach(s => {
        emotionCounts[s.emotion_label] = (emotionCounts[s.emotion_label] || 0) + 1;
    });

    const dominantStatsEl = document.getElementById('dominant-stats');
    dominantStatsEl.innerHTML = Object.entries(emotionCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([label, count]) => {
            const c = EMOTION_COLORS[label] || '#555';
            return `<span class="emotion-tag" style="background:${c}18;color:${c};">
                ${label.charAt(0).toUpperCase() + label.slice(1)}: ${count}
            </span>`;
        }).join('');

    // Quick Stats
    document.getElementById('n-segments').textContent = speechSegs.length;
    document.getElementById('audio-duration').textContent = `${summary.audio_duration_s.toFixed(1)}s`;
    const avgConf = speechSegs.reduce((s, x) => s + x.confidence, 0) / speechSegs.length;
    document.getElementById('avg-confidence').textContent = `${(avgConf * 100).toFixed(1)}%`;

    // Dimension bars
    requestAnimationFrame(() => {
        updateDimensionBar('arousal', summary.avg_arousal);
        updateDimensionBar('dominance', summary.avg_dominance);
        updateDimensionBar('valence', summary.avg_valence);
    });

    // Show section FIRST so Plotly can compute container widths
    const section = document.getElementById('results-section');
    section.style.display = 'block';
    section.classList.remove('slide-in');
    void section.offsetWidth;
    section.classList.add('slide-in');

    // Render charts AFTER section is visible (fixes alignment)
    requestAnimationFrame(() => {
        renderPieChart('pie-chart', speechSegs);
        renderTimelineChart('timeline-chart', speechSegs);
        renderRadarChart('radar-chart', summary);

        // Waveform Emotion chart — use real peaks if available, else fallback
        if (audioPeaks && audioDuration > 0) {
            renderWaveformEmotionChart('emotion-confidence-chart', audioPeaks, audioDuration, speechSegs);
        } else {
            renderEmotionConfidenceChart('emotion-confidence-chart', speechSegs);
        }
    });

    // Table
    renderSegmentTable(speechSegs);

    setTimeout(() => {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        // Force resize all plotly charts to ensure correct width
        document.querySelectorAll('.js-plotly-plot').forEach(el => Plotly.Plots.resize(el));
    }, 400);

    setStatus('success', summary, speechSegs.length, avgConf);
    document.getElementById('analyze-btn').disabled = false;
    setTimeout(loadHistory, 1000);
}

function renderSegmentTable(speechSegs) {
    const tbody = document.getElementById('segment-tbody');
    tbody.innerHTML = '';
    speechSegs.forEach((s, i) => {
        const color = EMOTION_COLORS[s.emotion_label] || '#aaa';
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="color:var(--text-3);font-family:monospace;">${i + 1}</td>
            <td style="color:var(--text-3);font-family:monospace;">${s.time_s.toFixed(1)}s</td>
            <td class="emotion-cell"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:8px;vertical-align:middle;box-shadow:0 0 6px ${color};"></span>${s.emoji}&nbsp;${s.emotion_label.toUpperCase()}</td>
            <td><span class="badge" style="background:${color}18;color:${color};">${(s.confidence * 100).toFixed(1)}%</span></td>
            <td style="color:#f9a825;font-family:monospace;">${s.arousal.toFixed(3)}</td>
            <td style="color:#ab47bc;font-family:monospace;">${s.dominance.toFixed(3)}</td>
            <td style="color:#00e5ff;font-family:monospace;">${s.valence.toFixed(3)}</td>
        `;
        tbody.appendChild(tr);
    });
}

function updateDimensionBar(dim, val) {
    document.getElementById(`bar-${dim}`).style.width = `${(val * 100).toFixed(1)}%`;
    document.getElementById(`val-${dim}`).textContent = val.toFixed(3);
}

/* ═══════════════════════════════════════════════════════
   HISTORY
═══════════════════════════════════════════════════════ */
const EMOJI_MAP = {
    happy: '😊', sad: '😢', angry: '😠', fearful: '😨',
    surprised: '😲', neutral: '😐', disgust: '🤢', calm: '😌',
    fear: '😨', surprise: '😲',
};

async function loadHistory() {
    try {
        const r = await fetch('/api/sessions');
        const d = await r.json();
        const wrap = document.getElementById('history-body');

        if (!d.sessions?.length) {
            wrap.innerHTML = '<div class="empty-state"><i class="ph-duotone ph-folder-open" style="font-size:2rem;"></i><p>No sessions yet. Analyze audio to get started.</p></div>';
            return;
        }

        wrap.innerHTML = `
        <div style="overflow-x:auto;">
            <table class="log-table session-table">
                <thead><tr>
                    <th>Date/Time</th>
                    <th>Duration</th>
                    <th>Segments</th>
                    <th>Avg Arousal</th>
                    <th>Avg Dominance</th>
                    <th>Avg Valence</th>
                </tr></thead>
                <tbody>
                ${d.sessions.map(s => `
                    <tr class="history-row" onclick="loadSessionDetails('${s['UUID']}')" title="Click to view full report">
                        <td style="color:var(--text-2);">${s['Time']}</td>
                        <td style="color:var(--green);">${s['Dur.']}</td>
                        <td><span class="badge">${s['Points']}</span></td>
                        <td style="color:#f9a825;font-family:monospace;">${(s['A'] || 0).toFixed(2)}</td>
                        <td style="color:#ab47bc;font-family:monospace;">${(s['D'] || 0).toFixed(2)}</td>
                        <td style="color:#00e5ff;font-family:monospace;">${(s['V'] || 0).toFixed(2)}</td>
                    </tr>
                `).join('')}
                </tbody>
            </table>
        </div>`;
    } catch {
        document.getElementById('history-body').innerHTML =
            '<div class="empty-state"><i class="ph-duotone ph-warning" style="font-size:2rem;"></i><p>Could not connect to Database.</p></div>';
    }
}

async function loadSessionDetails(uuid) {
    try {
        setStatus('Loading session details...', 'info');
        const res = await fetch(`/api/emotions/${uuid}`);
        const data = await res.json();

        if (!data.data || !data.data.length) {
            setStatus('⚠️ No emotion segments found for this session.', 'error');
            return;
        }

        const speechSegs = data.data.map((log, i) => ({
            time_s: i * 0.5,
            emotion_label: log.emotion_label,
            arousal: log.arousal,
            dominance: log.dominance,
            valence: log.valence,
            confidence: log.confidence,
            is_speech: true,
            emoji: EMOJI_MAP[log.emotion_label.toLowerCase()] || '🎭'
        }));

        const labels = speechSegs.map(s => s.emotion_label);
        const modeMap = {};
        let maxCount = 0, dominant = labels[0];
        labels.forEach(l => {
            modeMap[l] = (modeMap[l] || 0) + 1;
            if (modeMap[l] > maxCount) { maxCount = modeMap[l]; dominant = l; }
        });

        const avg_a = speechSegs.reduce((s, x) => s + x.arousal, 0) / speechSegs.length;
        const avg_d = speechSegs.reduce((s, x) => s + x.dominance, 0) / speechSegs.length;
        const avg_v = speechSegs.reduce((s, x) => s + x.valence, 0) / speechSegs.length;

        const summary = {
            dominant_emotion: dominant,
            dominant_emoji: EMOJI_MAP[dominant.toLowerCase()] || '🎭',
            avg_arousal: avg_a,
            avg_dominance: avg_d,
            avg_valence: avg_v,
            audio_duration_s: speechSegs.length * 0.5
        };

        renderResults({ segments: speechSegs, summary });
        setStatus(`✅ Loaded historical session with ${speechSegs.length} segments.`, 'success');

    } catch (e) {
        setStatus(`❌ Failed to load session details: ${e.message}`, 'error');
    }
}

/* ═══════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════ */
function setStatus(typeOrMsg, summaryOrType, segCount, avgConf) {
    const el = document.getElementById('status-msg');

    // Rich banner for success
    if (typeOrMsg === 'success' && typeof summaryOrType === 'object') {
        const s = summaryOrType;
        const eColor = EMOTION_COLORS[s.dominant_emotion] || '#00e5ff';
        el.className = 'result-banner success';
        el.innerHTML = `
            <div class="banner-row">
                <div class="banner-icon"><i class="ph-fill ph-check-circle"></i></div>
                <div class="banner-text">
                    <div class="banner-title">Analysis Complete</div>
                    <div class="banner-detail">
                        <span><i class="ph ph-waveform"></i> ${segCount} segments</span>
                        <span><i class="ph ph-target"></i> ${(avgConf * 100).toFixed(0)}% avg confidence</span>
                        <span style="color:${eColor};"><i class="ph ph-smiley"></i> ${s.dominant_emotion.charAt(0).toUpperCase() + s.dominant_emotion.slice(1)}</span>
                    </div>
                </div>
            </div>
        `;
        el.style.display = 'block';
        return;
    }

    // Simple text status (info/error/loading)
    const msg = typeOrMsg;
    const type = summaryOrType || 'info';
    if (!msg) { el.style.display = 'none'; return; }
    el.className = `result-banner ${type}`;
    el.innerHTML = `<div class="banner-row"><div class="banner-text"><div class="banner-title">${msg}</div></div></div>`;
    el.style.display = 'block';
}
