/**
 * Deployment Decision Engine — app.js
 */

(function () {
  'use strict';

  function revealPage() {
    var page = document.querySelector('.page');
    if (page) page.style.visibility = 'visible';
  }

  revealPage();
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

    qsa('#sidebarNav .tab-nav').forEach(function (item) {
      item.classList.toggle('active', item.dataset.tab === target);
    });

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
    void section.offsetWidth;
    section.classList.add('page-entering');
    setTimeout(function () { section.classList.remove('page-entering'); }, 600);
  }

  window.setActiveTab = setActiveTab;

  /* ═══════════════════════════════════════════════════════
     5. SPLASH TRANSITION
  ═══════════════════════════════════════════════════════ */
  function initSplash() {
    var splashBtn = byId('splashStartBtn');
    var appShell = qs('.app-shell');

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
    qsa('#sidebarNav .tab-nav').forEach(function (item) {
      item.addEventListener('click', function () {
        setActiveTab(item.dataset.tab || 'home');
      });
    });

    qsa('[data-go-tab]').forEach(function (card) {
      card.addEventListener('click', function () {
        setActiveTab(card.getAttribute('data-go-tab') || 'home');
      });
    });
  }

  /* ═══════════════════════════════════════════════════════
     7. FORM HELPERS
  ═══════════════════════════════════════════════════════ */
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

  (function () {
    var opts = [
      { label: 'DDR3', value: 'ddr3' },
      { label: 'DDR4', value: 'ddr4' },
      { label: 'DDR5', value: 'ddr5' },
      { label: 'LPDDR4', value: 'lpddr4' },
      { label: 'LPDDR5', value: 'lpddr5' }
    ];
    var idx = 1;
    window.cycleDDR = function (dir) {
      idx = (idx + dir + opts.length) % opts.length;
      var lbl = byId('ddr-label');
      var inp = byId('dep-ram-ddr');
      if (lbl) lbl.textContent = opts[idx].label;
      if (inp) inp.value = opts[idx].value;
    };
  }());

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
     8. SETTINGS
  ═══════════════════════════════════════════════════════ */
  var _autoRefreshId = null;

  function applyAutoRefresh(enabled) {
    var tog = byId('settingsAutoRefreshToggle');
    if (tog) tog.classList.toggle('stt-toggle-off', !enabled);
  }

  window.toggleSettingsAutoRefresh = function () {
    var current = storeGet('ui_auto_refresh', 'false') === 'true';
    var next = !current;
    storeSet('ui_auto_refresh', String(next));
    applyAutoRefresh(next);
  };

  function initSettings() {
    var theme = storeGet('ui_theme', 'dark');
    applyTheme(theme);

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
  // API client is defined in index.html

  /* ═══════════════════════════════════════════════════════
     10. STATE HYDRATION

     Latest-response-wins via _hydrateSeq: every call
     increments the counter and captures its own sequence
     number. When the async response arrives, it is discarded
     if a newer call has since been initiated.

     autoNav=true  → navigate to the stage-appropriate tab.
                     Used only on the initial DOMContentLoaded
                     load so the user lands on the right tab.
     autoNav=false → refresh panel data only; never changes
                     the active tab. Used by auto-refresh and
                     post-action callbacks so the user is not
                     forcibly navigated away from their current
                     view.
  ═══════════════════════════════════════════════════════ */
  var _hydrateSeq = 0;

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

  /**
   * @contract Intentionally neutralised — no-op stub.
   * index.html owns all panel-text rendering via renderUnifiedResult().
   * Do NOT add logic here; doing so would create a second, conflicting
   * render path and cause display inconsistencies.
   */
  function syncPanelText(data) { return; }

  // F-06: renderIssuesTable removed — it was dead code; never called anywhere.
  // The live render path is setDiagnosticsRows() in index.html, which also has
  // the correct .operator and .message fallbacks that this function lacked.

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

  async function _noop_hydrateFromBackend(autoNav) { return; }

  /* ═══════════════════════════════════════════════════════
     11. MODEL UPLOAD
  ═══════════════════════════════════════════════════════ */
  var _storedFile = null;
  var _uploadInFlight = false;

  function initModelUpload() {
    // NOTE: Upload wiring (drag-drop, file input, XHR progress) is fully
    // handled by index.html, which also sets storedModelFile for runFullPipeline.
    // Registering a second set of listeners here on #modelUploadZone caused
    // duplicate POST /api/model/index requests on every drag-drop.
    // Neutralised — do not add any listeners here.
    return;
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
  ═══════════════════════════════════════════════════════ */
  /**
   * @contract Intentionally neutralised — always returns null.
   * index.html owns deployment-profile collection and passes the profile
   * directly to runFullPipeline().  Reading the form fields here would
   * duplicate that logic and could return stale or mismatched values.
   */
  function readDeploymentProfile() { return null; }

  /**
   * @contract Intentionally neutralised — no-op stub.
   * index.html attaches the profile-save listener directly on #profileSaveBtn.
   * Registering a second listener here would cause duplicate POST
   * /api/deployment/profile requests on every save click.
   */
  function initProfileSave() { return; }

  /* ═══════════════════════════════════════════════════════
     13. RUN ANALYSIS
     Single call to /api/analyze-and-decide. The endpoint
     updates APP_STATE atomically before returning, so the
     response already contains the final authoritative result.
     renderDecisionResult renders it directly; no follow-up
     state fetch is needed and none is issued.
  ═══════════════════════════════════════════════════════ */
  var _analyzeInFlight = false;

  /**
   * @contract Intentionally neutralised — no-op stub.
   * index.html owns analyzeBtn → runFullPipeline() → renderUnifiedResult().
   * Registering a second listener here would cause a duplicate
   * POST /api/analyze-and-decide request on every click.
   */
  function initRunAnalysis() {
    return;
  }

  /* renderDecisionResult removed — renderUnifiedResult() in index.html is the sole render path. */

  /* ═══════════════════════════════════════════════════════
     14. BOOTSTRAP
  ═══════════════════════════════════════════════════════ */
  document.addEventListener('DOMContentLoaded', function () {
    revealPage();
    initSplash();
    bindTabInteractions();
    initSettings();
    initModelUpload();
    // initProfileSave and initRunAnalysis intentionally omitted:
    // index.html is the sole orchestrator of profile save, hydration, and analysis.
  });

}());

