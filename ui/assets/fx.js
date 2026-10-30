/* Crowd Safety AI - browser-side polish.
 * Injected into the *parent* page as a <script> (see ui/fx.py) because
 * st.markdown() never executes scripts. Everything here is progressive
 * enhancement: if it fails, the Streamlit app still works.
 *
 *  1. ambient particle canvas (≈40 dots, 30 fps cap, pauses when hidden)
 *  2. cursor spotlight on .ci-card
 *  3. count-up for [data-count-to] (only when the run/key is new)
 *  4. custom video controls with event markers + playhead readout, driven by
 *     real per-frame records found in #ci-run-data
 */
(function () {
  var VERSION = 4;
  var W = window, D = document;
  if (W.__ciFx && W.__ciFx.v === VERSION) { W.__ciFx.refresh(); return; }
  if (W.__ciFx && W.__ciFx.destroy) { try { W.__ciFx.destroy(); } catch (e) {} }

  var reduce = W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var seen = {};            // count-up keys already animated this session
  var enabled = true;
  var raf = 0, last = 0, parts = [], canvas = null, ctx = null, obs = null, scheduled = false;

  /* ---------- 1. ambient canvas ---------- */
  function ensureCanvas() {
    var app = D.querySelector('.stApp');
    if (!app) return false;
    if (!canvas || !canvas.isConnected) {
      canvas = D.createElement('canvas');
      canvas.id = 'ci-fx-canvas';
      canvas.setAttribute('aria-hidden', 'true');
      app.insertBefore(canvas, app.firstChild);
      ctx = canvas.getContext('2d');
      resize();
    }
    return true;
  }
  function resize() {
    if (!canvas) return;
    var dpr = Math.min(W.devicePixelRatio || 1, 1.5);
    canvas.width = Math.floor(W.innerWidth * dpr);
    canvas.height = Math.floor(W.innerHeight * dpr);
    canvas.style.width = W.innerWidth + 'px';
    canvas.style.height = W.innerHeight + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var n = Math.min(44, Math.floor(W.innerWidth * W.innerHeight / 42000));
    parts = [];
    for (var i = 0; i < n; i++) parts.push({
      x: Math.random() * W.innerWidth, y: Math.random() * W.innerHeight,
      vx: (Math.random() - .5) * .18, vy: (Math.random() - .5) * .18,
      r: Math.random() * 1.2 + .5
    });
  }
  function frame(ts) {
    raf = 0;
    if (!enabled || reduce || D.hidden || !ctx) return;
    if (ts - last > 33) {
      last = ts;
      var w = W.innerWidth, h = W.innerHeight;
      ctx.clearRect(0, 0, w, h);
      for (var i = 0; i < parts.length; i++) {
        var p = parts[i];
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = w; else if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h; else if (p.y > h) p.y = 0;
        ctx.fillStyle = 'rgba(140,215,255,.55)';
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832); ctx.fill();
        for (var j = i + 1; j < parts.length; j++) {
          var q = parts[j], dx = p.x - q.x, dy = p.y - q.y, d2 = dx * dx + dy * dy;
          if (d2 < 9000) {
            ctx.strokeStyle = 'rgba(90,209,255,' + (0.16 * (1 - d2 / 9000)).toFixed(3) + ')';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
          }
        }
      }
    }
    raf = W.requestAnimationFrame(frame);
  }
  function startLoop() {
    if (enabled && !reduce && !raf && ensureCanvas()) raf = W.requestAnimationFrame(frame);
  }
  function setEnabled(on) {
    enabled = !!on;
    D.documentElement.setAttribute('data-ci-fx', enabled ? 'on' : 'off');
    if (canvas) canvas.style.display = enabled && !reduce ? '' : 'none';
    if (enabled) startLoop();
  }

  /* ---------- 2. spotlight ---------- */
  function onMove(e) {
    var c = e.target && e.target.closest ? e.target.closest('.ci-card') : null;
    if (!c) return;
    var r = c.getBoundingClientRect();
    c.style.setProperty('--mx', (e.clientX - r.left) + 'px');
    c.style.setProperty('--my', (e.clientY - r.top) + 'px');
  }

  /* ---------- 3. count-up ---------- */
  function fmt(v, dec) {
    return v.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec });
  }
  function countUps() {
    var els = D.querySelectorAll('[data-count-to]:not([data-counted])');
    els.forEach(function (el) {
      el.setAttribute('data-counted', '1');
      var to = parseFloat(el.getAttribute('data-count-to'));
      var dec = parseInt(el.getAttribute('data-dec') || '0', 10);
      var key = el.getAttribute('data-count-key') || '';
      if (isNaN(to)) return;
      if (reduce || (key && seen[key])) { el.textContent = fmt(to, dec); return; }
      if (key) seen[key] = 1;
      var t0 = null, dur = 900;
      (function step(ts) {
        if (t0 === null) t0 = ts;
        var k = Math.min((ts - t0) / dur, 1), e = k === 1 ? 1 : 1 - Math.pow(2, -10 * k);
        el.textContent = fmt(to * e, dec);
        if (k < 1) W.requestAnimationFrame(step); else el.textContent = fmt(to, dec);
      })(performance.now());
    });
  }

  /* ---------- 4. video controls ---------- */
  function runData() {
    var el = D.getElementById('ci-run-data');
    if (!el) return null;
    var raw = el.getAttribute('data-json');
    if (el.__raw === raw) return el.__parsed;
    try { el.__parsed = JSON.parse(raw); el.__raw = raw; } catch (e) { el.__parsed = null; }
    return el.__parsed;
  }
  function mmss(s) {
    if (!isFinite(s)) s = 0;
    var m = Math.floor(s / 60), r = s - m * 60;
    return (m < 10 ? '0' : '') + m + ':' + (r < 10 ? '0' : '') + r.toFixed(1);
  }
  var ICON = {
    play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>',
    pause: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 2h3.2v12H3.5zM9.3 2h3.2v12H9.3z"/></svg>',
    prev: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 2h2v12H3zM14 2.5v11L6 8z"/></svg>',
    next: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M11 2h2v12h-2zM2 2.5v11L10 8z"/></svg>',
    full: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 2h5v2H4v3H2zM9 2h5v5h-2V4H9zM2 9h2v3h3v2H2zM12 9h2v5H9v-2h3z"/></svg>'
  };
  function btn(label, html, extra) {
    var b = D.createElement('button');
    b.type = 'button'; b.className = 'ci-vbtn' + (extra ? ' ' + extra : '');
    b.setAttribute('aria-label', label); b.title = label; b.innerHTML = html;
    return b;
  }
  function bindVideo() {
    var stage = D.querySelector('.st-key-video_stage');
    if (!stage) return;
    var video = stage.querySelector('video');
    if (!video) return;
    var data = runData() || {};
    var sig = (data.run_id || '') + '|' + (video.currentSrc || video.src || '');
    var bar = stage.querySelector('.ci-vbar');
    if (video.__ciSig === sig && bar && bar.isConnected) return;
    if (bar) bar.remove();
    var oldRead = stage.querySelector('.ci-readout'); if (oldRead) oldRead.remove();
    video.__ciSig = sig;
    (video.__ciOffs || []).forEach(function (o) { video.removeEventListener(o[0], o[1]); });
    var offs = video.__ciOffs = [];
    function on(type, fn) { video.addEventListener(type, fn); offs.push([type, fn]); }
    video.controls = false;
    video.setAttribute('playsinline', '');
    video.tabIndex = -1;

    var dur = function () { return isFinite(video.duration) && video.duration > 0 ? video.duration : (data.duration || 0); };
    var events = (data.events || []).filter(function (e) { return typeof e.s === 'number'; });

    bar = D.createElement('div'); bar.className = 'ci-vbar';
    var play = btn('Play or pause (space)', ICON.play);
    var jump = D.createElement('span'); jump.style.cssText = 'display:inline-flex;gap:6px';
    var prev = btn('Previous event marker', ICON.prev), next = btn('Next event marker', ICON.next);
    jump.appendChild(prev); jump.appendChild(next);
    var seek = D.createElement('div'); seek.className = 'ci-seek'; seek.tabIndex = 0;
    seek.setAttribute('role', 'slider'); seek.setAttribute('aria-label', 'Seek');
    seek.innerHTML = '<div class="ci-seek-track"><div class="ci-seek-fill"></div><div class="ci-seek-thumb"></div></div><div class="ci-seek-tip">00:00.0</div>';
    var track = seek.querySelector('.ci-seek-track'), fill = seek.querySelector('.ci-seek-fill'),
        thumb = seek.querySelector('.ci-seek-thumb'), tip = seek.querySelector('.ci-seek-tip');
    var time = D.createElement('span'); time.className = 'hide-sm'; time.textContent = '00:00.0 / 00:00.0';
    var speed = btn('Playback speed', '1×', 'hide-sm'); speed.style.minWidth = '48px';
    var full = btn('Toggle fullscreen (F)', ICON.full);
    [play, jump, seek, time, speed, full].forEach(function (n) { bar.appendChild(n); });
    stage.appendChild(bar);

    function drawTicks() {
      track.querySelectorAll('.ci-tick').forEach(function (t) { t.remove(); });
      var d = dur(); if (!d) return;
      events.forEach(function (e) {
        var t = D.createElement('span'); t.className = 'ci-tick ' + (e.sev || '');
        t.style.left = Math.min(100, e.s / d * 100) + '%';
        t.title = (e.label || 'Event') + ' @ ' + mmss(e.s);
        track.appendChild(t);
      });
    }
    drawTicks();
    on('loadedmetadata', function () { drawTicks(); paint(); });

    /* live readout driven by real records */
    var recs = data.records, readout = null;
    if (recs && recs.t && recs.t.length) {
      readout = D.createElement('div'); readout.className = 'ci-readout';
      readout.innerHTML =
        '<div><span>Frame</span><b data-live="frame">–</b></div>' +
        '<div><span>Density proxy</span><b data-live="density">–</b></div>' +
        '<div><span>Motion px/f</span><b data-live="motion">–</b></div>' +
        '<div><span>Risk index</span><b data-live="risk">–</b></div>' +
        '<div><span>Anomaly</span><b data-live="anomaly">–</b></div>';
      stage.appendChild(readout);
    }
    function idxAt(t) {
      var a = recs.t, lo = 0, hi = a.length - 1;
      while (lo < hi) { var m = (lo + hi + 1) >> 1; if (a[m] <= t) lo = m; else hi = m - 1; }
      return lo;
    }
    function setLive(k, txt, tone) {
      var n = readout.querySelector('[data-live="' + k + '"]');
      if (!n) return; n.textContent = txt; n.className = tone || '';
    }
    function paint() {
      var d = dur(), t = video.currentTime || 0, pct = d ? Math.min(100, t / d * 100) : 0;
      fill.style.width = pct + '%'; thumb.style.left = pct + '%';
      time.textContent = mmss(t) + ' / ' + mmss(d);
      seek.setAttribute('aria-valuemin', '0'); seek.setAttribute('aria-valuemax', String(Math.round(d)));
      seek.setAttribute('aria-valuenow', String(Math.round(t))); seek.setAttribute('aria-valuetext', mmss(t));
      if (readout) {
        var i = idxAt(t);
        setLive('frame', String(recs.f ? recs.f[i] : i + 1));
        setLive('density', recs.density[i].toFixed(3));
        setLive('motion', recs.motion[i].toFixed(2));
        var r = recs.risk[i];
        setLive('risk', r.toFixed(2), r > 0.7 ? 'crit' : r > 0.5 ? 'warn' : 'ok');
        var a = recs.anomaly[i];
        setLive('anomaly', a.toFixed(2), a > 0.7 ? 'warn' : '');
      }
    }
    var ticking = false;
    function loop() { paint(); if (!video.paused && !video.ended) W.requestAnimationFrame(loop); else ticking = false; }
    function setIcon() { play.innerHTML = video.paused ? ICON.play : ICON.pause; }
    function toggle() { if (video.paused) { var p = video.play(); if (p && p.catch) p.catch(function () {}); } else video.pause(); }
    on('play', function () { setIcon(); if (!ticking) { ticking = true; W.requestAnimationFrame(loop); } });
    on('pause', function () { setIcon(); paint(); });
    on('ended', function () { setIcon(); paint(); });
    on('timeupdate', function () { if (video.paused) paint(); });
    on('seeked', paint);
    on('click', toggle);
    play.addEventListener('click', toggle);

    function seekTo(t) { var d = dur(); video.currentTime = Math.max(0, Math.min(d || t, t)); paint(); }
    function pos(e) { var r = track.getBoundingClientRect(); return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)); }
    var dragging = false;
    seek.addEventListener('pointerdown', function (e) { dragging = true; seek.setPointerCapture(e.pointerId); seekTo(pos(e) * dur()); });
    seek.addEventListener('pointermove', function (e) {
      var p = pos(e), d = dur(), t = p * d, label = '';
      events.forEach(function (ev) { if (d && Math.abs(ev.s - t) / d < 0.012) label = ' · ' + (ev.label || 'event'); });
      tip.textContent = mmss(t) + label; tip.style.left = (p * 100) + '%';
      if (dragging) seekTo(t);
    });
    seek.addEventListener('pointerup', function () { dragging = false; });
    seek.addEventListener('keydown', function (e) {
      var fps = data.fps || 25, step = e.shiftKey ? 1 / fps : 1;
      if (e.key === 'ArrowRight') { seekTo(video.currentTime + step); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { seekTo(video.currentTime - step); e.preventDefault(); }
      else if (e.key === 'Home') { seekTo(0); e.preventDefault(); }
      else if (e.key === 'End') { seekTo(dur()); e.preventDefault(); }
      else if (e.key === ' ' || e.key === 'k') { toggle(); e.preventDefault(); }
      else if (e.key === 'f') { full.click(); e.preventDefault(); }
    });
    function jumpEv(dir) {
      var t = video.currentTime, list = events.map(function (e) { return e.s; }).sort(function (a, b) { return a - b; }), tgt = null;
      if (dir > 0) { for (var i = 0; i < list.length; i++) if (list[i] > t + 0.05) { tgt = list[i]; break; } }
      else { for (var j = list.length - 1; j >= 0; j--) if (list[j] < t - 0.4) { tgt = list[j]; break; } }
      if (tgt !== null) seekTo(tgt);
    }
    prev.addEventListener('click', function () { jumpEv(-1); });
    next.addEventListener('click', function () { jumpEv(1); });
    if (!events.length) { prev.disabled = next.disabled = true; prev.style.opacity = next.style.opacity = .35; prev.title = next.title = 'No event markers for this run'; }
    var rates = [1, 1.5, 2, 0.5], ri = 0;
    speed.addEventListener('click', function () { ri = (ri + 1) % rates.length; video.playbackRate = rates[ri]; speed.textContent = rates[ri] + '×'; });
    full.addEventListener('click', function () {
      if (D.fullscreenElement) D.exitFullscreen(); else if (stage.requestFullscreen) stage.requestFullscreen();
    });
    paint(); setIcon();
  }

  /* ---------- observer / lifecycle ---------- */
  function scan() {
    scheduled = false;
    try { countUps(); bindVideo(); ensureCanvas(); } catch (e) { if (W.console) console.warn('[ci-fx]', e); }
  }
  function schedule() { if (!scheduled) { scheduled = true; W.requestAnimationFrame(scan); } }

  var api = {
    v: VERSION,
    setEnabled: setEnabled,
    refresh: schedule,
    destroy: function () {
      if (obs) obs.disconnect();
      if (raf) W.cancelAnimationFrame(raf);
      D.removeEventListener('pointermove', onMove);
      W.removeEventListener('resize', onResize);
      D.removeEventListener('visibilitychange', startLoop);
      if (canvas && canvas.remove) canvas.remove();
    }
  };
  function onResize() { resize(); }
  D.addEventListener('pointermove', onMove, { passive: true });
  W.addEventListener('resize', onResize);
  D.addEventListener('visibilitychange', startLoop);
  var app = D.querySelector('.stApp') || D.body;
  obs = new MutationObserver(schedule);
  obs.observe(app, { childList: true, subtree: true });
  W.__ciFx = api;
  schedule();
  startLoop();
})();
