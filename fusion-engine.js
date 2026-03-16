/* =============================================================================
   CamGuard AI — Advanced Multi-Sensor Fusion Engine  (fusion-engine.js)
   Version: 2.0  |  Professional Surveillance Detection System
   =============================================================================
   Architecture:
     • FusionEngine  : Weighted risk scorer across all sensor channels
     • NeuralMatcher : Pattern-matching against stored fingerprints
     • NetFingerprint: Deep ONVIF / RTSP / ESP32 / XMeye fingerprinting
     • ThresholdAdapt: Self-improving detection thresholds from feedback
   ============================================================================= */

'use strict';

// ── Weights for each sensor channel (must sum to 1.0) ──────────────────────
const FUSION_WEIGHTS = {
    visual: 0.35,
    lens: 0.25,
    network: 0.20,
    magnetic: 0.15,
    ir: 0.05
};

// ── Pre-defined Neural Signature Fingerprints ──────────────────────────────
// Grouped signatures for known surveillance hardware.
// magSignature: typical μT reading range [min, max]
// aiConfidence : typical COCO/YOLO confidence floor
const SURVEILLANCE_FINGERPRINTS = [
    // ── Hidden / Spy Cameras ──────────────────────────────────────────────────
    { label: 'Spy Cam (Motion Trigger)', category: 'surveillance', risk: 'critical', magSignature: [112, 124], aiConfidence: 0.72, networkHints: ['esp', 'espressif', 'tuya'] },
    { label: 'IP Wi-Fi Pinhole Chipset', category: 'surveillance', risk: 'critical', magSignature: [138, 147], aiConfidence: 0.80, networkHints: ['onvif', 'rtsp', 'ipcam'] },
    { label: 'IR Night-Vision Module', category: 'surveillance', risk: 'critical', magSignature: [183, 194], aiConfidence: 0.75, networkHints: ['hikvision', 'dahua', 'xmeye'] },
    { label: 'Button / Pinhole Lens CPU', category: 'surveillance', risk: 'high', magSignature: [90, 99], aiConfidence: 0.65, networkHints: [] },
    { label: 'Hidden Mic Transmitter', category: 'surveillance', risk: 'high', magSignature: [68, 76], aiConfidence: 0.55, networkHints: [] },
    { label: 'ESP32-CAM Module', category: 'surveillance', risk: 'critical', magSignature: [105, 115], aiConfidence: 0.70, networkHints: ['esp32', 'espressif', 'ai-thinker'] },
    { label: 'HiSilicon SoC Camera', category: 'surveillance', risk: 'critical', magSignature: [130, 145], aiConfidence: 0.82, networkHints: ['hisilicon', 'hisi', 'xmeye'] },
    { label: 'Tuya Smart Camera', category: 'surveillance', risk: 'high', magSignature: [95, 110], aiConfidence: 0.68, networkHints: ['tuya', 'smartlife'] },
    { label: 'ONVIF Surveillance Device', category: 'surveillance', risk: 'critical', magSignature: [120, 155], aiConfidence: 0.85, networkHints: ['onvif', 'device_service'] },
    { label: 'RTSP Stream Source', category: 'surveillance', risk: 'high', magSignature: [100, 160], aiConfidence: 0.80, networkHints: ['rtsp', '554', '8554'] },

    // ── Reference / Non-Threat Devices ─────────────────────────────────────
    { label: 'Smartphone', category: 'reference', risk: 'low', magSignature: [40, 55], aiConfidence: 0.90, networkHints: [] },
    { label: 'Bluetooth Speaker', category: 'reference', risk: 'low', magSignature: [305, 340], aiConfidence: 0.91, networkHints: [] },
    { label: 'Laptop / Notebook', category: 'reference', risk: 'medium', magSignature: [128, 142], aiConfidence: 0.88, networkHints: [] },
    { label: 'Wall Power Outlet / Wiring', category: 'reference', risk: 'low', magSignature: [540, 570], aiConfidence: 0.30, networkHints: [] },
    { label: 'AC Adapter / Transformer', category: 'reference', risk: 'low', magSignature: [405, 425], aiConfidence: 0.30, networkHints: [] },
    { label: 'Electric Fan / Motor', category: 'reference', risk: 'low', magSignature: [200, 280], aiConfidence: 0.50, networkHints: [] },
];

// ── Advanced Network Surveillance Keywords ─────────────────────────────────
const ADVANCED_NET_SIGNATURES = {
    critical: [
        'onvif', 'rtsp', 'device_service',
        'hikvision', 'dahua', 'xmeye', 'hisi', 'hisilicon',
        'esp32', 'espressif', 'ai-thinker',
        'tuya', 'smartlife', 'iot_camera',
        'mjpg-streamer', 'ipcamera', 'netcam',
        'dvr', 'nvr', 'cctv', 'surveillance'
    ],
    high: [
        'webcam', 'ipcam', 'ip camera', 'network camera',
        'video server', 'camera http', 'cam interface',
        'foscam', 'reolink', 'amcrest', 'annke',
        'wyze', 'blink', 'arlo', 'ring cam', 'nest cam'
    ],
    medium: [
        'axis', 'bosch', 'avigilon', 'vivotek', 'hanwha', 'uniview', 'tiandy',
        'flir', 'mobotix', 'milestone', 'genetec'
    ]
};

// ── IoT Camera-Specific URL Probes ─────────────────────────────────────────
const IOT_PROBE_PATHS = [
    '/onvif/device_service',          // ONVIF standard
    '/cgi-bin/snapshot.cgi',          // Dahua / Generic
    '/ISAPI/Security/userCheck',      // Hikvision
    '/web/index.html',                // Axis WebUI
    '/mjpg/video.mjpg',               // MJPEG stream
    '/stream',                        // ESP32-CAM
    '/?action=stream',                // Generic action cam
    '/video.cgi',                     // Foscam
    '/cgi-bin/magicBox.cgi',          // Dahua API
    '/tmpfs/auto.jpg',                // HiSilicon snapshot
    '/snapshot.jpg',                  // Generic snapshot
];

/* =============================================================================
   FUSION ENGINE — Central Risk Scoring
   Receives per-module scores and combines using weighted formula.
   ============================================================================= */
const FusionEngine = {
    /**
     * Compute final camera probability from all sensor channels.
     * @param {object} scores  { visual, lens, network, magnetic, ir }  all in [0..100]
     * @returns {{ fused: number, breakdown: object, label: string, risk: string }}
     */
    computeRisk(scores)
    {
        const s = {
            visual: Math.min(100, scores.visual || 0),
            lens: Math.min(100, scores.lens || 0),
            network: Math.min(100, scores.network || 0),
            magnetic: Math.min(100, scores.magnetic || 0),
            ir: Math.min(100, scores.ir || 0),
        };

        const fused = Math.round(
            s.visual * FUSION_WEIGHTS.visual +
            s.lens * FUSION_WEIGHTS.lens +
            s.network * FUSION_WEIGHTS.network +
            s.magnetic * FUSION_WEIGHTS.magnetic +
            s.ir * FUSION_WEIGHTS.ir
        );

        const clamped = Math.min(100, fused);

        // Multi-channel bonuses: if 3+ channels confirm, bump score
        const activeChannels = Object.values(s).filter(v => v >= 25).length;
        const bonus = activeChannels >= 3 ? 10 : activeChannels >= 2 ? 5 : 0;
        const finalScore = Math.min(100, clamped + bonus);

        const risk = finalScore >= 75 ? 'critical'
            : finalScore >= 50 ? 'high'
                : finalScore >= 25 ? 'medium'
                    : 'low';

        const label = finalScore >= 75 ? '🔴 CAMERA DETECTED'
            : finalScore >= 50 ? '🟠 HIGH RISK'
                : finalScore >= 25 ? '🟡 SUSPICIOUS'
                    : '🟢 CLEAR';

        return { fused: finalScore, breakdown: s, risk, label, activeChannels };
    },

    /**
     * Render a live Fusion Panel into a given container element.
     */
    renderPanel(containerId, result)
    {
        const el = document.getElementById(containerId);
        if (!el) return;
        const { fused, breakdown, risk, label, activeChannels } = result;
        const barColor = risk === 'critical' ? '#ef4444' : risk === 'high' ? '#f59e0b' : risk === 'medium' ? '#a855f7' : '#10b981';
        el.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <span style="font-size:0.8rem;font-weight:700;color:${barColor}">${label}</span>
        <span style="font-size:0.95rem;font-weight:800;color:${barColor}">${fused}%</span>
      </div>
      <div style="height:6px;border-radius:3px;background:rgba(255,255,255,0.08);margin-bottom:14px;overflow:hidden">
        <div style="height:100%;width:${fused}%;background:${barColor};border-radius:3px;transition:width 0.4s ease"></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:0.65rem">
        ${FusionEngine._row('🤖 Visual AI', breakdown.visual, FUSION_WEIGHTS.visual)}
        ${FusionEngine._row('🔦 Lens Reflection', breakdown.lens, FUSION_WEIGHTS.lens)}
        ${FusionEngine._row('📡 Network', breakdown.network, FUSION_WEIGHTS.network)}
        ${FusionEngine._row('🧲 Magnetic', breakdown.magnetic, FUSION_WEIGHTS.magnetic)}
        ${FusionEngine._row('🌡️ IR Sensor', breakdown.ir, FUSION_WEIGHTS.ir)}
      </div>
      <div style="margin-top:10px;font-size:0.6rem;color:var(--text-muted);text-align:center">
        ${activeChannels} active sensor${activeChannels !== 1 ? 's' : ''} · Weighted Fusion Score
      </div>`;
    },

    _row(label, score, weight)
    {
        const pct = Math.round(score);
        const color = pct >= 70 ? '#f87171' : pct >= 40 ? '#fbbf24' : '#6ee7b7';
        return `<div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:6px 8px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span style="color:var(--text-muted)">${label}</span>
        <span style="color:${color};font-weight:700">${pct}%</span>
      </div>
      <div style="height:3px;border-radius:2px;background:rgba(255,255,255,0.08)">
        <div style="height:100%;width:${pct}%;background:${color};border-radius:2px;transition:width 0.3s"></div>
      </div>
      <div style="color:rgba(255,255,255,0.3);margin-top:2px">weight: ${Math.round(weight * 100)}%</div>
    </div>`;
    }
};

/* =============================================================================
   NEURAL MATCHER — Fingerprint-Based Pattern Recognition
   Compares real-time sensor data against stored + pre-defined signatures.
   ============================================================================= */
const NeuralMatcher = {
    /**
     * Match a live mag reading against the full signature library.
     * Returns the best matching fingerprint or null.
     * @param {number} magValue  Live μT reading
     * @param {string[]} networkHints  Device info strings from network scan
     * @param {object} db  The live local database
     */
    findMatch(magValue, networkHints = [], db)
    {
        let bestMatch = null;
        let bestScore = 0;

        const allFingerprints = [
            ...SURVEILLANCE_FINGERPRINTS,
            ...db.knowledgeBase.filter(k => k.magSignature)
        ];

        for (const fp of allFingerprints) {
            let score = 0;

            // Magnetic range match
            if (fp.magSignature) {
                const [lo, hi] = fp.magSignature;
                if (magValue >= lo && magValue <= hi) {
                    score += 60;
                } else {
                    const mid = (lo + hi) / 2;
                    const range = (hi - lo) / 2 + 15; // tolerance band
                    const dist = Math.abs(magValue - mid);
                    if (dist < range) score += Math.round((1 - dist / range) * 40);
                }
            }

            // Network hint match (partial string match)
            if (networkHints.length && fp.networkHints?.length) {
                const netStr = networkHints.join(' ').toLowerCase();
                const netMatches = fp.networkHints.filter(h => netStr.includes(h));
                score += netMatches.length * 20;
            }

            if (score > bestScore) {
                bestScore = score;
                bestMatch = { ...fp, matchScore: Math.min(100, score) };
            }
        }

        // Only return if confidence is meaningful
        return bestScore >= 40 ? bestMatch : null;
    },

    /**
     * Match network fingerprint strings against the advanced signature DB.
     * Returns { risk, label, matchedTerms }
     */
    matchNetworkSignature(htmlContent = '', hostname = '')
    {
        const text = (htmlContent + ' ' + hostname).toLowerCase();
        const matchedTerms = [];

        let highestRisk = 'low';
        const riskRank = { critical: 3, high: 2, medium: 1, low: 0 };

        for (const [risk, terms] of Object.entries(ADVANCED_NET_SIGNATURES)) {
            for (const term of terms) {
                if (text.includes(term)) {
                    matchedTerms.push(term);
                    if (riskRank[risk] > riskRank[highestRisk]) highestRisk = risk;
                }
            }
        }

        const score = highestRisk === 'critical' ? 95
            : highestRisk === 'high' ? 70
                : highestRisk === 'medium' ? 45 : 0;

        return {
            risk: highestRisk,
            score,
            matchedTerms,
            isSurveillance: highestRisk === 'critical' || highestRisk === 'high'
        };
    },

    /**
     * Get the IoT probe paths array for deep fingerprinting.
     */
    getIotProbePaths() { return IOT_PROBE_PATHS; }
};

/* =============================================================================
   THRESHOLD ADAPTOR — Self-Improving Detection Thresholds
   ============================================================================= */
const ThresholdAdaptor = {
    /**
     * Called after a user confirms OR rejects a detection.
     * Nudges all adaptive thresholds in the right direction.
     */
    adapt(db, confirmed, falseAlarm)
    {
        const total = db.confirmedDetections + db.falseAlarms;
        const accuracy = total > 0 ? db.confirmedDetections / total : 0.5;

        if (falseAlarm) {
            // Less sensitive: raise thresholds
            db.adaptedGlintRatio = Math.min(2.8, db.adaptedGlintRatio + 0.06);
            db.adaptedRiskThreshold = Math.min(60, db.adaptedRiskThreshold + 2);
            db.adaptedAIThreshold = Math.min(0.85, db.adaptedAIThreshold + 0.02);
            db.adaptedFusionThreshold = Math.min(75, (db.adaptedFusionThreshold || 50) + 3);
        }

        if (confirmed && !falseAlarm) {
            // More sensitive: lower thresholds
            db.adaptedGlintRatio = Math.max(1.2, db.adaptedGlintRatio - 0.04);
            db.adaptedRiskThreshold = Math.max(18, db.adaptedRiskThreshold - 2);
            db.adaptedAIThreshold = Math.max(0.40, db.adaptedAIThreshold - 0.02);
            db.adaptedFusionThreshold = Math.max(25, (db.adaptedFusionThreshold || 50) - 3);
        }

        // Smooth convergence: pull thresholds toward accuracy-derived ideal
        const idealFusion = Math.round(80 - accuracy * 40); // 40%–80%
        const current = db.adaptedFusionThreshold || 50;
        db.adaptedFusionThreshold = Math.round(current * 0.9 + idealFusion * 0.1);

        console.log(`[ThresholdAdaptor] accuracy=${(accuracy * 100).toFixed(1)}% | fusion=${db.adaptedFusionThreshold}% | ai=${db.adaptedAIThreshold.toFixed(2)} | glint=${db.adaptedGlintRatio.toFixed(2)}`);

        return db;
    },

    /**
     * Store a detailed detection pattern into the knowledge base.
     */
    storePattern(db, {
        label, aiConfidence, magValue, networkInfo,
        visualScore, lensScore, fusedScore, confirmed, environment
    })
    {
        const entry = {
            label: label || 'unknown',
            aiConfidence: aiConfidence || 0,
            magValue: magValue || null,
            magSignature: magValue ? [magValue - 4, magValue + 4] : null,
            networkInfo: networkInfo || null,
            visualScore: visualScore || 0,
            lensScore: lensScore || 0,
            fusedScore: fusedScore || 0,
            confirmed: !!confirmed,
            category: confirmed ? 'surveillance' : 'unconfirmed',
            risk: fusedScore >= 75 ? 'critical' : fusedScore >= 50 ? 'high' : 'medium',
            environment: environment || {},
            timestamp: Date.now()
        };

        db.knowledgeBase.push(entry);

        // Keep ring buffer at max 500
        if (db.knowledgeBase.length > 500) db.knowledgeBase.shift();

        return entry;
    },

    /** Compute human-readable accuracy stats */
    getStats(db)
    {
        const total = db.confirmedDetections + db.falseAlarms;
        const accuracy = total > 0 ? (db.confirmedDetections / total * 100).toFixed(1) : 'N/A';
        const kb = db.knowledgeBase.length;
        const surv = db.knowledgeBase.filter(k => k.category === 'surveillance').length;
        return { accuracy, kb, surv, total };
    }
};

/* =============================================================================
   ADVANCED NETWORK FINGERPRINTER
   Deep probe of discovered devices using IoT surveillance paths.
   ============================================================================= */
const AdvancedNetFingerprinter = {
    /**
     * Deeply probe an IP for surveillance-specific endpoints.
     * Returns { detected, deviceLabel, risk, score, matchedPaths, matchedTerms }
     */
    async probe(ip, openPorts = [])
    {
        const paths = NeuralMatcher.getIotProbePaths();
        const portList = openPorts.length ? openPorts : [80, 8080, 554, 8554];
        const matchedPaths = [];
        let combinedText = '';

        for (const port of portList.slice(0, 4)) { // probe max 4 ports
            for (const path of paths.slice(0, 5)) {  // probe max 5 paths
                try {
                    const url = `http://${ip}:${port}${path}`;
                    const ctrl = new AbortController();
                    const timer = setTimeout(() => ctrl.abort(), 1200);
                    const resp = await fetch(url, { mode: 'no-cors', signal: ctrl.signal });
                    clearTimeout(timer);
                    // CORS blocks reading text, but response status / type is useful
                    if (resp.type === 'opaque' || resp.ok) matchedPaths.push(`${port}${path}`);
                    combinedText += ` ${path} status:${resp.status}`;
                } catch (_) {
                    // Timeout or CORS — path might still exist (opaque)
                }
            }
        }

        const sigResult = NeuralMatcher.matchNetworkSignature(combinedText, ip);

        return {
            detected: sigResult.isSurveillance || matchedPaths.length > 0,
            deviceLabel: sigResult.matchedTerms[0] || (matchedPaths.length ? 'IP Camera' : 'Unknown'),
            risk: sigResult.risk,
            score: Math.min(100, sigResult.score + matchedPaths.length * 15),
            matchedPaths,
            matchedTerms: sigResult.matchedTerms
        };
    }
};

/* =============================================================================
   PUBLIC API — Export to global scope (used by app.js)
   ============================================================================= */
window.CamGuardFusion = {
    FusionEngine,
    NeuralMatcher,
    ThresholdAdaptor,
    AdvancedNetFingerprinter,
    SURVEILLANCE_FINGERPRINTS,
    FUSION_WEIGHTS,

    /**
     * Main entry point: compute full fused risk from all sensor data.
     * @param {object} scores   { visual, lens, network, magnetic, ir } (0–100 each)
     * @param {object} meta     { magValue, networkHints, networkDevices, aiDetections }
     * @param {object} db       The live persistend database
     * @returns Full result with fused score, matched fingerprint, and UI-ready data.
     */
    analyze(scores, meta = {}, db)
    {
        const fusionResult = FusionEngine.computeRisk(scores);
        const neuralMatch = NeuralMatcher.findMatch(
            meta.magValue || 0,
            meta.networkHints || [],
            db
        );

        const threshold = db.adaptedFusionThreshold || 50;
        const isAlert = fusionResult.fused >= threshold;

        return {
            ...fusionResult,
            neuralMatch,
            isAlert,
            threshold,
            confidence: {
                camera: fusionResult.fused + '%',
                magnetic: neuralMatch ? neuralMatch.label : 'No Match',
                network: meta.networkDevices > 0 ? 'YES' : 'NO',
                lens: scores.lens >= 30 ? 'DETECTED' : 'CLEAR',
                ir: scores.ir >= 20 ? 'ACTIVE' : 'NONE'
            }
        };
    }
};
