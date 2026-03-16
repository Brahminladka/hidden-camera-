// ================================================================
// MULTI-SENSOR FUSION PANEL FUNCTIONS
// ================================================================

/** Open the Fusion Panel and immediately compute a result */
function openFusionPanel()
{
    openScreen('screen-fusion');
    computeFusionNow();
    updateFusionThresholdDisplay();
}

/** Compute current fusion score using all gathered sensor scores */
function computeFusionNow()
{
    if (!window.CamGuardFusion) {
        showToast('❌ Fusion Engine not loaded', 'danger');
        return;
    }

    const scores = {
        visual: state.scores.visual || 0,
        lens: state.scores.lens || 0,
        network: state.scores.network || 0,
        magnetic: state.scores.magnetic || 0,
        ir: state.scores.ir || 0
    };

    const meta = {
        magValue: state.fusionMeta.magValue || state.magnetic.lastValue || 0,
        networkHints: state.fusionMeta.networkHints || [],
        networkDevices: state.fusionMeta.networkDevices || state.network.devices.length || 0
    };

    const result = window.CamGuardFusion.analyze(scores, meta, db);
    state.fusionMeta.lastFusionResult = result;

    // Render the weighted bar chart panel
    window.CamGuardFusion.FusionEngine.renderPanel('fusionPanel', result);

    // Populate confidence report rows
    const setEl = (id, val, color) =>
    {
        const el = document.getElementById(id);
        if (el) { el.textContent = val; if (color) el.style.color = color; }
    };

    const riskColor = result.risk === 'critical' ? '#ef4444'
        : result.risk === 'high' ? '#f59e0b'
            : result.risk === 'medium' ? '#a855f7' : '#10b981';

    setEl('fc-camera', result.confidence.camera, riskColor);
    setEl('fc-magnetic', result.confidence.magnetic,
        result.neuralMatch?.category === 'surveillance' ? '#f87171' : '#34d399');
    setEl('fc-network', result.confidence.network,
        result.confidence.network === 'YES' ? '#f87171' : '#34d399');
    setEl('fc-lens', result.confidence.lens,
        result.confidence.lens === 'DETECTED' ? '#f59e0b' : '#34d399');
    setEl('fc-ir', result.confidence.ir,
        result.confidence.ir === 'ACTIVE' ? '#f59e0b' : '#34d399');

    const nm = result.neuralMatch;
    setEl('fc-neural',
        nm ? `🦾 ${nm.label} (${nm.matchScore}% match)` : '✅ No Surveillance Match',
        nm?.category === 'surveillance' ? '#f87171' : '#34d399');

    updateFusionThresholdDisplay();

    // Auto-toast if alert threshold crossed
    if (result.isAlert) {
        showToast(`🚨 FUSION ALERT: ${result.label} — ${result.fused}% probability!`, 'danger', 6000);
    }
}

/** Update the self-learning threshold display in Fusion Panel */
function updateFusionThresholdDisplay()
{
    const stats = window.CamGuardFusion?.ThresholdAdaptor.getStats(db);
    if (!stats) return;
    const s = id => document.getElementById(id);
    if (s('ft-fusion')) s('ft-fusion').textContent = `${db.adaptedFusionThreshold || 50}%`;
    if (s('ft-ai')) s('ft-ai').textContent = `${(db.adaptedAIThreshold * 100).toFixed(0)}%`;
    if (s('ft-glint')) s('ft-glint').textContent = db.adaptedGlintRatio.toFixed(2);
    if (s('ft-kb')) s('ft-kb').textContent = `${stats.kb} patterns (${stats.surv} surveillance)`;
    if (s('ft-confirmed')) s('ft-confirmed').textContent = stats.total;
    if (s('ft-accuracy')) s('ft-accuracy').textContent = stats.accuracy === 'N/A' ? '—' : `${stats.accuracy}%`;
}

/** Reset all sensor scores to zero */
function resetFusionPanel()
{
    state.scores = { visual: 0, lens: 0, network: 0, magnetic: 0, ir: 0 };
    state.fusionMeta = { magValue: 0, networkHints: [], networkDevices: 0, lastFusionResult: null };
    computeFusionNow();
    showToast('↺ Fusion scores reset', 'info', 2000);
}

// ── Cross-sensor score updaters (called from each scan module) ─────────────

function updateIRFusionScore(intensity)
{
    state.scores.ir = Math.min(100, Math.round(intensity * 1.2));
    state.findings.ir = intensity > 25 ? 'IR Active' : 'Clear';
}

function updateMagFusionScore(pct, isSurveillanceMatch)
{
    state.scores.magnetic = Math.min(100, isSurveillanceMatch ? pct + 30 : pct);
    state.fusionMeta.magValue = state.magnetic.lastValue;
}

function updateVisualFusionScore(camCount, maxConf)
{
    state.scores.visual = Math.min(100, camCount * 30 + Math.round(maxConf * 40));
}

function updateLensFusionScore(riskPct)
{
    state.scores.lens = Math.min(100, riskPct);
}

function updateNetFusionScore(devices)
{
    state.scores.network = Math.min(100, devices.filter(d => d.isSuspicious).length * 35);
    state.fusionMeta.networkDevices = devices.length;
    state.fusionMeta.networkHints = devices
        .flatMap(d => [d.vendor, d.deviceName])
        .filter(Boolean)
        .map(s => s.toLowerCase());
}
