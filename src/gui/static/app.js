/**
 * Deployment Decision Engine — app.js
 *
 * This file provides robust UI initialization that runs independently
 * of the backend. It:
 *  1. Immediately makes .page visible (bypassing any visibility:hidden issue)
 *  2. Sets up all tab navigation
 *  3. Wires up the splash "Get Started" flow
 *  4. Initialises GPU toggles, arch/DDR cyclers, steppers
 *  5. Manages theme (dark/light) persistence
 *  6. Provides the API client and state hydration
 *  7. Drives the Analyze pipeline and result rendering
 *
 * Include AFTER the inline styles in <head> and BEFORE </body>:
 *   <script src="/static/app.js"></script>
 * (Remove or keep the large inline <script> block — if both are present
 *  the inline block's window-scoped helpers will coexist safely.)
 */

(function () {
  'use strict';

  /* ═══════════════════════════════════════════════════════
     0. IMMEDIATE PAGE REVEAL
     The .page div has visibility:hidden in CSS to prevent a
     flash of un-styled content. We reveal it the moment this
     script runs — no backend round-trip needed.
  ═══════════════════════════════════════════════════════ */
  function revealPage() {
    var page = document.querySelector('.page');
    if (page) page.style.visibility = 'visible';
  }

  // Try immediately (script may run after DOM is parsed)
  revealPage();
  // Guarantee on DOMContentLoaded
  document.addEventListener('DOMContentLoaded', revealPage);

  /* ═══════════════════════════════════════════════════════
     1. UTILITIES
  ═══════════════════════════════════════════════════════ */
  function byId(id) { return document.getElementById(id); }
  function qs(sel, ctx) { return (ctx || document).querySelector(sel); }
  function qsa(sel, ctx) { return Array.from((ctx || document).querySelectorAll(sel)); }

  function safeNum(v, min, max) {
    min = min == null ? 0 : min;
    max = max == null ? 100 : max;
    var n = Number(v);
    if (!isFinite(n)) return null;
    return Math.max(min, Math.min(max, n));
  }

  function setText(id, value) {
    var el = byId(id);
    if (!el) return;
    var hasValue = value != null && value !== '' && value !== '--';
    el.textContent = hasValue ? String(value) : '--';
  }

  function setPercent(id, value) {
    var el = byId(id);
    if (!el) return;
    var p = safeNum(value);
    el.textContent = p == null ? '--' : Math.round(p) + '%';
  }

  function setBar(id, value) {
    var el = byId(id);
    if (!el) return;
    var p = safeNum(value);
    el.style.width = (p == null ? 0 : p) + '%';
  }

  /* ═══════════════════════════════════════════════════════
     2. STORAGE HELPERS
  ═══════════════════════════════════════════════════════ */
  var _store = (function () {
    try {
      localStorage.setItem('__t', '1');
      localStorage.removeItem('__t');
      return localStorage;
    } catch (e) { return null; }
  }());

  function storeGet(key, fallback) {
    if (!_store) return fallback;
    var v = _store.getItem(key);
    return v == null ? fallback : v;
  }

  function storeSet(key, value) {
    if (_store) _store.setItem(key, value);
  }

  /* ═══════════════════════════════════════════════════════
     3. THEME
  ═══════════════════════════════════════════════════════ */
  function applyTheme(theme) {
    var isLight = theme === 'light';
    document.body.classList.toggle('light-mode', isLight);

    var themeBtn = byId('themeBtn');
    if (themeBtn) themeBtn.textContent = isLight ? '🌙' : '☀️';

    var settingsToggle = byId('settingsThemeToggle');
    if (settingsToggle) {
      settingsToggle.classList.toggle('stt-toggle-off', !isLight);
    }

    var chip = qs('#stt-theme-row .stt-chip');
    if (chip) chip.textContent = isLight ? 'Light' : 'Dark';
  }

  function toggleTheme() {
    var next = document.body.classList.contains('light-mode') ? 'dark' : 'light';
    storeSet('ui_theme', next);
    applyTheme(next);
  }

  window.toggleTheme = toggleTheme;

  /* ═══════════════════════════════════════════════════════
     4. TAB / PAGE NAVIGATION
  ═══════════════════════════════════════════════════════ */
  var TAB_IDS = ['home', 'model', 'analysis', 'reports', 'settings'];

  function setActiveTab(tabName) {
    var target = TAB_IDS.indexOf(tabName) >= 0 ? tabName : 'home';

    // Cross-fade: exit current visible section, enter target
    var currentSection = qs('.page-section:not(.hidden)');
    var nextSection = byId('page-' + target);

    if (currentSection && currentSection !== nextSection) {
      currentSection.classList.add('page-exiting');
      setTimeout(function () {
        currentSection.classList.add('hidden');
        currentSection.classList.remove('active', 'page-exiting');
        showSection(nextSection);
      }, 200);
    } else {
      showSection(nextSection);
    }

    // Update sidebar nav active state
    qsa('#sidebarNav .tab-nav').forEach(function (item) {
      item.classList.toggle('active', item.dataset.tab === target);
    });

    // Sync nav pill/bar indicator
    var pillTarget = qs('#sidebarNav .tab-nav[data-tab="' + target + '"]');
    if (pillTarget) {
      requestAnimationFrame(function () {
        if (window.__movePill) window.__movePill(pillTarget);
        if (window.__moveBar) window.__moveBar(pillTarget);
      });
    }
  }

  function showSection(section) {
    if (!section) return;
    section.classList.remove('hidden');
    section.classList.add('active');
    section.classList.remove('page-entering');
    void section.offsetWidth; // reflow trigger
    section.classList.add('page-entering');
    setTimeout(function () { section.classList.remove('page-entering'); }, 600);
  }

  window.setActiveTab = setActiveTab;

  /* ═══════════════════════════════════════════════════════
     5. SPLASH → HOME DASHBOARD TRANSITION
  ═══════════════════════════════════════════════════════ */
  function initSplash() {
    var splashBtn = byId('splashStartBtn');
    var appShell = qs('.app-shell');

    // Ensure splash mode is active initially
    if (appShell) appShell.classList.add('splash-mode');

    if (splashBtn) {
      splashBtn.addEventListener('click', function () {
        var splashEl = byId('page-home');
        var splashView = byId('splashView');
        var homeDash = byId('homeDashboard');

        if (splashEl) splashEl.classList.add('splash-exit');

        setTimeout(function () {
          if (appShell) appShell.classList.remove('splash-mode');
          if (splashEl) {
            splashEl.classList.remove('splash-exit');
            splashEl.classList.remove('splash-section');
            splashEl.style.display = '';
          }
          if (splashView) splashView.classList.add('hidden');
          if (homeDash) homeDash.classList.remove('hidden');
        }, 550);
      });
    }
  }

  /* ═══════════════════════════════════════════════════════
     6. TAB INTERACTION WIRING
  ═══════════════════════════════════════════════════════ */
  function bindTabInteractions() {
    // Sidebar nav items
    qsa('#sidebarNav .tab-nav').forEach(function (item) {
      item.addEventListener('click', function () {
        setActiveTab(item.dataset.tab || 'home');
      });
    });

    // Cards with data-go-tab
    qsa('[data-go-tab]').forEach(function (card) {
      card.addEventListener('click', function () {
        setActiveTab(card.getAttribute('data-go-tab') || 'home');
      });
    });
  }

  /* ═══════════════════════════════════════════════════════
     7. FORM HELPERS  (steppers, arch, DDR cyclers, GPU)
  ═══════════════════════════════════════════════════════ */

  // Numeric stepper
  window.stepNum = function (id, delta) {
    var el = byId(id);
    if (!el) return;
    var cur = parseFloat(el.value) || 0;
    var min = parseFloat(el.min) || 0;
    var max = parseFloat(el.max) || Infinity;
    var step = parseFloat(el.step) || 1;
    el.value = Math.min(max, Math.max(min, cur + delta * step));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };

  // Architecture cycler
  (function () {
    var opts = ['x86', 'ARM', 'ARM64', 'RISC-V'];
    var idx = 0;
    window.cycleArch = function (dir) {
      idx = (idx + dir + opts.length) % opts.length;
      var lbl = byId('arch-label');
      var inp = byId('dep-cpu-arch');
      if (lbl) lbl.textContent = opts[idx];
      if (inp) inp.value = opts[idx].toLowerCase();
    };
  }());

  // DDR type cycler
  (function () {
    var opts = [
      { label: 'DDR3', value: 'ddr3' },
      { label: 'DDR4', value: 'ddr4' },
      { label: 'DDR5', value: 'ddr5' },
      { label: 'LPDDR4', value: 'lpddr4' },
      { label: 'LPDDR5', value: 'lpddr5' }
    ];
    var idx = 1; // start at DDR4
    window.cycleDDR = function (dir) {
      idx = (idx + dir + opts.length) % opts.length;
      var lbl = byId('ddr-label');
      var inp = byId('dep-ram-ddr');
      if (lbl) lbl.textContent = opts[idx].label;
      if (inp) inp.value = opts[idx].value;
    };
  }());

  // GPU card toggles
  window.gpuToggle = function (checkId, cardId) {
    var check = byId(checkId);
    var card = byId(cardId);
    var pill = byId('gpill-' + checkId);
    if (!check || !card || !pill) return;
    check.checked = !check.checked;
    var on = check.checked;
    card.classList.toggle('gpu-on', on);
    pill.classList.toggle('gpu-pill-on', on);
    var lbl = pill.querySelector('.gpu-pill-label');
    if (lbl) lbl.textContent = on ? 'ON' : 'OFF';
    check.dispatchEvent(new Event('change', { bubbles: true }));
  };

  /* ═══════════════════════════════════════════════════════
     8. SETTINGS — AUTO REFRESH + THEME TOGGLE
  ═══════════════════════════════════════════════════════ */
  var _autoRefreshId = null;

  function startAutoRefresh() {
    if (_autoRefreshId) return;
    _autoRefreshId = setInterval(function () {
      hydrateFromBackend().catch(function () {});
    }, 10000);
  }

  function stopAutoRefresh() {
    if (_autoRefreshId) { clearInterval(_autoRefreshId); _autoRefreshId = null; }
  }

  function applyAutoRefresh(enabled) {
    var tog = byId('settingsAutoRefreshToggle');
    if (tog) tog.classList.toggle('stt-toggle-off', !enabled);
    enabled ? startAutoRefresh() : stopAutoRefresh();
  }

  window.toggleSettingsAutoRefresh = function () {
    var current = storeGet('ui_auto_refresh', 'false') === 'true';
    var next = !current;
    storeSet('ui_auto_refresh', String(next));
    applyAutoRefresh(next);
  };

  function initSettings() {
    // Theme
    var theme = storeGet('ui_theme', 'dark');
    applyTheme(theme);

    // Theme toggle button
    var themeBtn = byId('themeBtn');
    if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

    var settingsThemeTog = byId('settingsThemeToggle');
    if (settingsThemeTog) {
      settingsThemeTog.addEventListener('click', toggleTheme);
    }

    var sttThemeRow = byId('stt-theme-row');
    if (sttThemeRow) {
      sttThemeRow.addEventListener('click', function (e) {
        if (!e.target.closest('#settingsThemeToggle')) toggleTheme();
      });
    }

    // Auto refresh
    var autoEnabled = storeGet('ui_auto_refresh', 'false') === 'true';
    applyAutoRefresh(autoEnabled);

    var autoTog = byId('settingsAutoRefreshToggle');
    if (autoTog) {
      autoTog.addEventListener('click', window.toggleSettingsAutoRefresh);
    }
  }

  /* ═══════════════════════════════════════════════════════
     9. API CLIENT
  ═══════════════════════════════════════════════════════ */
  async function apiFetch(url, options) {
    options = options || {};
    options.credentials = 'same-origin';
    var resp = await fetch(url, options);
    var data = null;
    try { data = await resp.json(); } catch (_) {}
    if (!resp.ok) {
      var msg = (data && (data.detail || data.error)) || ('HTTP ' + resp.status);
      throw new Error(msg);
    }
    return data;
  }

  var API = {
    state:    function () { return apiFetch('/api/state'); },
    profile:  function (payload) {
      return apiFetch('/api/deployment/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    },
    deleteProfile: function () {
      return fetch('/api/deployment/profile', { method: 'DELETE', credentials: 'same-origin' });
    },
    index:    function (fd) {
      return fetch('/api/model/index', { method: 'POST', body: fd, credentials: 'same-origin' })
        .then(function (r) { return r.json(); });
    },
    diagnostics: function () {
      return apiFetch('/api/model/diagnostics', { method: 'POST' });
    },
    analyze:  function () {
      return apiFetch('/api/model/analyze', { method: 'POST' });
    },
    decision: function (params) {
      params = params || {};
      var qs = [];
      if (params.debug) qs.push('debug=true');
      if (params.explain) qs.push('explain=true');
      var suffix = qs.length ? '?' + qs.join('&') : '';
      return apiFetch('/api/decision/recommend' + suffix, { method: 'POST' });
    },
    analyzeAndDecide: function (payload) {
      return apiFetch('/api/analyze-and-decide-by-id', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    },
    gpuCalibration: function (fd) {
      return fetch('/api/calibration/gpu/load', { method: 'POST', body: fd, credentials: 'same-origin' })
        .then(function (r) { return r.json(); });
    }
  };

  /* ═══════════════════════════════════════════════════════
     10. STATE HYDRATION
  ═══════════════════════════════════════════════════════ */
  function setAnalyzeEnabled(enabled, hint) {
    var btn = byId('analyzeBtn');
    if (btn) btn.disabled = !enabled;
    var hintEl = byId('modelRunHint');
    if (hintEl) {
      hintEl.textContent = enabled
        ? 'Ready · Click to run the full analysis pipeline'
        : (hint || 'Upload a model file and save a Deployment Profile to enable analysis');
    }
  }

  function syncPanelText(data) {
    data = data || {};
    setText('rt-runtimeTop', data.runtimeTop);
    setText('rt-runtimeSub', data.runtimeSub);
    setText('rt-runtimeScore', data.runtimeScore);
    setText('rt-decisionLabel', data.decision);
    setText('rt-latencyValue', data.latency);
    setText('rt-precisionText', data.precision);
    setText('rt-backendStatus', data.backendStatus);
    setText('rt-confidenceVerdict', data.confVerdict);
    setPercent('rt-confidencePercent', data.confScore);
    setBar('rt-opsCoverageBar', data.covPct);

    setText('conf-backendStatus', data.backendStatus);
    setText('conf-confidenceVerdict', data.confVerdict);
    setText('conf-decisionLabel', data.decision);
    setText('conf-latencyValue', data.latency);
    setText('conf-precisionText', data.precision);
    setPercent('conf-confidencePercent', data.confScore);

    // Update donut arc
    var arc = byId('conf-donutArc');
    if (arc) arc.style.strokeDashoffset = String(data.confScore != null ? 138.2 * (1 - data.confScore / 100) : 138.2);

    setText('compat-backendStatus', data.backendStatus);
    setText('compat-confidenceVerdict', data.confVerdict);
    setText('compat-precisionText', data.precision);
    setPercent('compat-confidencePercent', data.confScore);
    setBar('compat-opsCoverageBar', data.covPct);

    var scoreEl = byId('sb-scoreLabel');
    if (scoreEl) {
      scoreEl.textContent = data.confScore != null
        ? Math.round(data.confScore) + '%' : '--';
    }

    // Model name
    if (data.modelName) setText('conf-modelName', data.modelName);
  }

  function renderIssuesTable(rows) {
    var table = byId('issuesTable');
    if (!table) return;
    var safeRows = Array.isArray(rows) ? rows : [];
    if (safeRows.length === 0) {
      table.innerHTML = '<div style="padding:24px;text-align:center;color:rgba(200,170,255,0.35);font-size:12px;">No issues detected</div>';
      return;
    }
    function sevIcon(s) {
      var sl = String(s || '').toLowerCase();
      if (sl === 'high')   return { icon: '⚠️', bg: 'rgba(248,113,113,0.12)', cls: 'sev-high', label: 'High' };
      if (sl === 'med' || sl === 'medium') return { icon: '◐', bg: 'rgba(251,191,36,0.12)', cls: 'sev-med', label: 'Med' };
      return { icon: '✓', bg: 'rgba(52,211,153,0.12)', cls: 'sev-low', label: 'Low' };
    }
    table.innerHTML = safeRows.map(function(d) {
      var s = sevIcon(d.severity);
      return '<div class="tbl-row">' +
        '<div class="tbl-node"><div class="tbl-icon" style="background:' + s.bg + ';">' + s.icon + '</div>' +
        '<span class="tbl-name">' + String(d.node || '--') + '</span></div>' +
        '<div class="tbl-problem">' + String(d.problem || '--') + '</div>' +
        '<div style="text-align:center;"><span class="sev-pill ' + s.cls + '">' + s.label + '</span></div>' +
        '<div class="tbl-runtime">' + String(d.runtime || 'All') + '</div>' +
        '</div>';
    }).join('');
  }

  function resetPanels() {
    syncPanelText({
      runtimeTop: '--', runtimeSub: '--', runtimeScore: '--',
      decision: '--', latency: '--', backendStatus: '--',
      confVerdict: '--', confScore: null, covPct: null
    });
  }

  function stageToTab(stage) {
    if (stage === 'decision' || stage === 'analysis' || stage === 'diagnostics') return 'analysis';
    if (stage === 'model') return 'model';
    return 'home';
  }

  async function hydrateFromBackend() {
    try {
      var stateData = await API.state();
      var stage = String(stateData && stateData.stage || 'empty');

      setActiveTab(stageToTab(stage));
      revealPage(); // ensure visible after hydration

      if (stage === 'empty' || !stateData.model) {
        resetPanels();
        setAnalyzeEnabled(false, 'Upload a model file to get started');
        return;
      }

      var modelPath = String(stateData.model && stateData.model.path || '');
      var modelBase = modelPath ? modelPath.split(/[/\\]/).pop() : '--';
      setText('modelName', modelBase || '--');
      var stageEl = byId('sb-stageLabel');
      if (stageEl) stageEl.textContent = 'Loaded';

      var profileExists = stateData.deployment_profile != null;
      var modelExists = stateData.model != null;
      // Enable Run Analysis only when BOTH model and profile are present
      // (server requires diagnostics stage first, but we run the full pipeline)
      setAnalyzeEnabled(profileExists && modelExists,
        !modelExists ? 'Upload a model file to enable analysis' :
        'Save a Deployment Profile to enable analysis');

      // Populate panels from state
      // Real /api/state shape:
      //   analysis: { evaluations:[{runtime,decision,predicted_latency_ms,confidence_score,...}], best_runtime, confidence }
      //   decision: { runtime, status, confidence, risk_level, reasons }
      var analysis = stateData.analysis;
      var decision = stateData.decision;

      if (decision || analysis) {
        var bestRuntime = (decision && decision.runtime) ||
                          (analysis && analysis.best_runtime) || '--';

        var evals = (analysis && analysis.evaluations) || [];
        var bestEval = evals.find(function (e) {
          return String(e.runtime || '').toUpperCase() === String(bestRuntime).toUpperCase();
        }) || evals[0] || null;

        // Confidence: decision gives final adjusted value; analysis gives raw score
        var confRaw = (decision && decision.confidence != null) ? decision.confidence
                    : (analysis && analysis.confidence != null) ? analysis.confidence : null;
        var confPct = confRaw != null
          ? safeNum(confRaw <= 1 ? confRaw * 100 : confRaw) : null;

        var statusStr = (decision && decision.status) ||
                        (bestEval && bestEval.decision) || '--';
        var riskLevel = (decision && decision.risk_level) || null;

        var latency = bestEval && bestEval.predicted_latency_ms != null
          ? Number(bestEval.predicted_latency_ms).toFixed(1) + ' ms' : '--';

        syncPanelText({
          runtimeTop:    String(bestRuntime).toLowerCase(),
          runtimeSub:    'recommended engine',
          runtimeScore:  confPct != null ? Math.round(confPct) + '%' : '--',
          decision:      statusStr,
          latency:       latency,
          backendStatus: riskLevel || (confPct != null
            ? (confPct >= 80 ? 'HIGH' : confPct >= 60 ? 'MEDIUM' : 'LOW') : '--'),
          confVerdict:   statusStr === 'SUPPORTED' ? 'Supported' :
                         statusStr === 'CONDITIONAL' ? 'Conditional' : statusStr,
          confScore:     confPct,
          covPct:        bestEval ? safeNum((bestEval.utility_score || 0) * 100) : null
        });

        // Populate issues table from decision reasons or eval diagnostics
        var reasons = (decision && Array.isArray(decision.reasons)) ? decision.reasons : [];
        var diagRows = reasons.map(function (r) {
          return { node: bestRuntime, problem: String(r),
                   severity: statusStr === 'REJECTED' ? 'high' : 'med', runtime: bestRuntime };
        });
        if (diagRows.length === 0 && bestEval && Array.isArray(bestEval.diagnostics)) {
          diagRows = bestEval.diagnostics.map(function (d) {
            return { node: bestRuntime, problem: String(d), severity: 'med', runtime: bestRuntime };
          });
        }
        if (diagRows.length > 0) renderIssuesTable(diagRows);
      }
    } catch (err) {
      console.warn('Could not reach backend:', err.message || err);
      revealPage(); // still reveal the page even without backend
    }
  }

  /* ═══════════════════════════════════════════════════════
     11. MODEL UPLOAD
     Uses /api/model/index (the real endpoint in gui_app.py)
  ═══════════════════════════════════════════════════════ */
  var _storedFile = null; // keep the File object so we can re-send for analysis

  function initModelUpload() {
    var zone      = byId('modelUploadZone');
    var fileInput = byId('modelFileInput');

    function handleFile(file) {
      if (!file) return;
      if (!file.name.toLowerCase().endsWith('.onnx')) {
        alert('Only ONNX model files (.onnx) are supported.');
        return;
      }
      _storedFile = file;
      var fd = new FormData();
      fd.append('file', file);

      setModelStatus('uploading', file.name);

      fetch('/api/model/index', { method: 'POST', body: fd, credentials: 'same-origin' })
        .then(function (r) {
          if (!r.ok) return r.json().then(function(e){ throw new Error(e.detail || ('HTTP ' + r.status)); });
          return r.json();
        })
        .then(function (data) {
          if (data && data.success !== false) {
            setModelStatus('loaded', file.name);
          } else {
            var errMsg = (data && data.error) || 'Upload/analysis failed';
            setModelStatus('error', errMsg);
            _storedFile = null;
          }
          hydrateFromBackend();
        })
        .catch(function (err) {
          setModelStatus('error', 'Upload failed: ' + (err.message || err));
          _storedFile = null;
        });
    }

    if (fileInput) {
      fileInput.addEventListener('change', function () {
        if (fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
      });
    }

    if (zone) {
      zone.addEventListener('dragover',  function (e) { e.preventDefault(); zone.classList.add('drag-over'); });
      zone.addEventListener('dragleave', function ()  { zone.classList.remove('drag-over'); });
      zone.addEventListener('drop', function (e) {
        e.preventDefault();
        zone.classList.remove('drag-over');
        var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) handleFile(f);
      });
      zone.addEventListener('click', function () { if (fileInput) fileInput.click(); });
    }
  }

  function setModelStatus(status, name) {
    var nameEl  = byId('modelName');
    var stageEl = byId('sb-stageLabel');
    var hintEl  = byId('modelRunHint');
    if (status === 'uploading') {
      if (nameEl)  nameEl.textContent  = 'Uploading…';
      if (stageEl) stageEl.textContent = 'Uploading';
      if (hintEl)  hintEl.textContent  = 'Uploading model, please wait…';
    } else if (status === 'loaded') {
      if (nameEl)  nameEl.textContent  = name || '--';
      if (stageEl) stageEl.textContent = 'Loaded';
      if (hintEl)  hintEl.textContent  = 'Model ready. Save a Deployment Profile to run analysis.';
    } else if (status === 'error') {
      if (nameEl)  nameEl.textContent  = 'Error';
      if (stageEl) stageEl.textContent = 'Error';
      if (hintEl)  hintEl.textContent  = name || 'Upload failed';
    }
  }

  /* ═══════════════════════════════════════════════════════
     12. DEPLOYMENT PROFILE SAVE
     POST /api/deployment/profile with JSON matching DeploymentProfileRequest
  ═══════════════════════════════════════════════════════ */
  function readDeploymentProfile() {
    function val(id)           { var el = byId(id); return el ? el.value : null; }
    function checked(id)       { var el = byId(id); return el ? el.checked : false; }
    function num(id, fallback) { var v = parseFloat(val(id)); return isFinite(v) ? v : fallback; }
    return {
      cpu_cores:          num('dep-cpu-cores', 4),
      cpu_arch:           val('dep-cpu-arch') || 'x86',
      gpu_available:      checked('dep-gpu-avail'),
      cuda_available:     checked('dep-cuda-avail'),
      trt_available:      checked('dep-trt-avail'),
      vram_gb:            num('dep-vram-gb', 0),
      ram_gb:             num('dep-ram-gb', 8),
      ram_ddr:            val('dep-ram-ddr') || 'ddr4',
      target_latency_ms:  num('dep-target-latency', 100),
      memory_limit_mb:    num('dep-memory-limit', 2048),
    };
  }

  function initProfileSave() {
    var saveBtn = byId('saveProfileBtn');
    if (!saveBtn) return;
    saveBtn.addEventListener('click', function () {
      var profile = readDeploymentProfile();
      saveBtn.disabled = true;
      var origText = saveBtn.textContent;
      saveBtn.textContent = 'Saving…';
      API.profile(profile)
        .then(function () {
          saveBtn.textContent = 'Saved ✓';
          setTimeout(function () {
            saveBtn.disabled = false;
            saveBtn.textContent = origText;
          }, 2000);
          hydrateFromBackend();
        })
        .catch(function (err) {
          saveBtn.disabled = false;
          saveBtn.textContent = origText;
          alert('Failed to save profile: ' + (err.message || err));
        });
    });
  }

  /* ═══════════════════════════════════════════════════════
     13. RUN ANALYSIS
     The real server uses a 3-step pipeline:
       1. POST /api/model/diagnostics  (no body needed — uses APP_STATE)
       2. POST /api/model/analyze      (no body needed — uses APP_STATE)
       3. POST /api/decision/recommend (no body needed — uses APP_STATE)
     All three must be called in sequence after upload + profile are set.
  ═══════════════════════════════════════════════════════ */
  function initRunAnalysis() {
    var btn = byId('analyzeBtn');
    if (!btn) return;

    btn.addEventListener('click', function () {
      var hintEl = byId('modelRunHint');
      function setHint(msg) { if (hintEl) hintEl.textContent = msg; }
      function fail(msg) {
        setHint('Error: ' + msg);
        alert('Analysis failed: ' + msg);
        btn.disabled = false;
        btn.innerHTML = origHtml;
      }

      // Must have stored file from upload
      if (!_storedFile) {
        alert('No model file found — please re-upload your model.');
        return;
      }

      btn.disabled = true;
      var origHtml = btn.innerHTML;
      btn.innerHTML = '<svg width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" stroke-dasharray="40 20" transform-origin="center"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/></circle></svg> Running…';
      setHint('Sending model for analysis…');

      // Use /api/analyze-and-decide — single endpoint, takes file + form fields
      var profile = readDeploymentProfile();
      var fd = new FormData();
      fd.append('file', _storedFile);
      fd.append('cpu_cores',        String(profile.cpu_cores        || 1));
      fd.append('ram_gb',           String(profile.ram_gb           || 8));
      fd.append('gpu_available',    String(!!profile.gpu_available));
      fd.append('cuda_available',   String(!!profile.cuda_available));
      fd.append('vram_gb',          String(profile.vram_gb          || 0));
      fd.append('trt_available',    String(!!profile.trt_available));
      fd.append('cpu_arch',         String(profile.cpu_arch         || 'x86'));
      fd.append('ram_ddr',          String(profile.ram_ddr          || 'ddr4'));
      fd.append('target_latency_ms',String(profile.target_latency_ms|| 100));
      fd.append('memory_limit_mb',  String(profile.memory_limit_mb  || 2048));

      fetch('/api/analyze-and-decide', {
        method: 'POST',
        body: fd,
        credentials: 'same-origin'
      })
      .then(function (r) {
        return r.json().then(function (data) {
          if (!r.ok) throw new Error((data && (data.detail || data.error)) || ('HTTP ' + r.status));
          return data;
        });
      })
      .then(function (data) {
        btn.disabled = false;
        btn.innerHTML = origHtml;
        renderDecisionResult(data);
        setHint('Analysis complete.');
        setActiveTab('analysis');
        hydrateFromBackend();
      })
      .catch(function (err) {
        fail(err.message || String(err));
      });
    });
  }

  function renderDecisionResult(data) {
    // /api/analyze-and-decide returns:
    // { success, summary, runtime_details:{runtime,status,confidence,...},
    //   evaluations:[{runtime,decision,predicted_latency_ms,...}],
    //   analysis_state:{best_runtime, confidence}, risk_score, risk_level }
    if (!data || !data.success) {
      var errMsg = (data && data.detail) || 'Unknown error';
      alert('Analysis returned an error: ' + errMsg);
      return;
    }

    var rd        = data.runtime_details || {};
    var as_       = data.analysis_state  || {};
    var evals     = data.evaluations     || [];
    var summary   = data.summary         || {};

    var runtime   = rd.runtime   || as_.best_runtime || '--';
    var status    = rd.status    || '--';
    var riskLevel = data.risk_level || '--';

    var confRaw = rd.confidence != null ? rd.confidence : as_.confidence;
    var confPct = confRaw != null ? safeNum(confRaw <= 1 ? confRaw * 100 : confRaw) : null;

    var bestEval = evals.find(function (e) {
      return String(e.runtime || '').toUpperCase() === String(runtime).toUpperCase();
    }) || evals[0] || null;

    var latency = (summary.estimated_latency_ms != null)
      ? Number(summary.estimated_latency_ms).toFixed(1) + ' ms'
      : (bestEval && bestEval.predicted_latency_ms != null)
        ? Number(bestEval.predicted_latency_ms).toFixed(1) + ' ms' : '--';

    syncPanelText({
      runtimeTop:   String(runtime).toLowerCase(),
      runtimeSub:   'recommended engine',
      runtimeScore: confPct != null ? Math.round(confPct) + '%' : '--',
      decision:     status,
      latency:      latency,
      precision:    bestEval ? (bestEval.precision_support || '--') : '--',
      backendStatus: riskLevel,
      confVerdict:  status === 'SUPPORTED'    ? 'Supported'   :
                    status === 'CONDITIONAL'  ? 'Conditional' :
                    status === 'REJECTED'     ? 'Rejected'    : status,
      confScore:    confPct,
      covPct:       bestEval ? safeNum((bestEval.utility_score || 0) * 100) : null
    });

    // Issues from evaluations diagnostics
    var issueRows = [];
    evals.forEach(function (ev) {
      if (!Array.isArray(ev.diagnostics)) return;
      ev.diagnostics.forEach(function (msg) {
        issueRows.push({ node: ev.runtime || '--', problem: String(msg),
          severity: ev.decision === 'UNSUPPORTED' ? 'high' : 'med', runtime: ev.runtime || 'All' });
      });
    });
    // Also show rejection reasons from runtime_details
    if (Array.isArray(rd.reasons)) {
      rd.reasons.forEach(function (r) {
        issueRows.push({ node: runtime, problem: String(r),
          severity: status === 'REJECTED' ? 'high' : 'med', runtime: runtime });
      });
    }
    renderIssuesTable(issueRows);
  }


  /* ═══════════════════════════════════════════════════════
     14. BOOTSTRAP
  ═══════════════════════════════════════════════════════ */
  document.addEventListener('DOMContentLoaded', function () {
    revealPage();
    initSplash();
    bindTabInteractions();
    initSettings();
    initModelUpload();
    initProfileSave();
    initRunAnalysis();
    hydrateFromBackend();
  });

}());
