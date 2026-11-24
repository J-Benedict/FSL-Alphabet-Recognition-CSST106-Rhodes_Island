(() => {
  const cfg = window.FSL_CONFIG;
  const video = document.getElementById("webcam");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const placeholder = document.getElementById("cameraPlaceholder");
  const toggleCamBtn = document.getElementById("toggleCamBtn");
  const captureNowBtn = document.getElementById("captureNowBtn");

  const modePhraseBtn = document.getElementById("modePhraseBtn");
  const modeSpellBtn = document.getElementById("modeSpellBtn");
  const openSignGuideBtn = document.getElementById("openSignGuideBtn");
  const langToggleBtn = document.getElementById("langToggleBtn");
  const langBadgeText = document.getElementById("langBadgeText");

  const statusBanner = document.getElementById("statusBanner");
  const claheBadge = document.getElementById("claheBadge");
  const latencyHud = document.getElementById("latencyHud");
  const fpsHud = document.getElementById("fpsHud");
  const dwellRingHud = document.getElementById("dwellRingHud");
  const dwellRingProgress = document.getElementById("dwellRingProgress");
  const RING_CIRCUMFERENCE = 2 * Math.PI * 22;

  const settingsGearBtn = document.getElementById("settingsGearBtn");
  const settingsPopover = document.getElementById("settingsPopover");
  const stageMirrorBtn = document.getElementById("stageMirrorBtn");
  const modelSelect = document.getElementById("modelSelect");
  const cameraSelect = document.getElementById("cameraSelect");
  const autoSpeakToggle = document.getElementById("autoSpeakToggle");
  const autoSpaceToggle = document.getElementById("autoSpaceToggle");
  const autoClaheToggle = document.getElementById("autoClaheToggle");
  const claheToggle = document.getElementById("claheToggle");
  const skeletonToggle = document.getElementById("skeletonToggle");
  const mirrorToggle = document.getElementById("mirrorToggle");

  const detectedBox = document.getElementById("detectedBox");
  const modeTag = document.getElementById("modeTag");
  const detectedSub = document.getElementById("detectedSub");
  const detectedLetter = document.getElementById("detectedLetter");
  const thresholdTag = document.getElementById("thresholdTag");
  const top3List = document.getElementById("top3List");
  const quickPickRow = document.getElementById("quickPickRow");

  const composerHint = document.getElementById("composerHint");
  const composerTokens = document.getElementById("composerTokens");
  const naturalPreview = document.getElementById("naturalPreview");
  const autocompleteRow = document.getElementById("autocompleteRow");
  const sendSpeakBtn = document.getElementById("sendSpeakBtn");
  const spaceBtn = document.getElementById("spaceBtn");
  const undoBtn = document.getElementById("undoBtn");
  const clearComposerBtn = document.getElementById("clearComposerBtn");

  const signGuideModal = document.getElementById("signGuideModal");
  const signGuideBackdrop = document.getElementById("signGuideBackdrop");
  const signGuideClose = document.getElementById("signGuideClose");
  const guideTabPhrases = document.getElementById("guideTabPhrases");
  const guideTabAlphabet = document.getElementById("guideTabAlphabet");
  const guidePhrasesPane = document.getElementById("guidePhrasesPane");
  const guideAlphabetPane = document.getElementById("guideAlphabetPane");
  const guideSearchInput = document.getElementById("guideSearchInput");
  const guideCategorySelect = document.getElementById("guideCategorySelect");
  const guidePhraseGrid = document.getElementById("guidePhraseGrid");
  const guideDetailCard = document.getElementById("guideDetailCard");

  const refModal = document.getElementById("refModal");
  const refModalBackdrop = document.getElementById("refModalBackdrop");
  const refModalClose = document.getElementById("refModalClose");
  const refModalBadge = document.getElementById("refModalBadge");
  const refModalTitle = document.getElementById("refModalTitle");
  const refModalImg = document.getElementById("refModalImg");
  const refModalCounter = document.getElementById("refModalCounter");
  const refPrevBtn = document.getElementById("refPrevBtn");
  const refNextBtn = document.getElementById("refNextBtn");
  const refSampleThumbs = Array.from(document.querySelectorAll(".ref-sample-thumb"));
  const cheatTiles = Array.from(document.querySelectorAll(".cheat-tile"));

  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = 640;
  captureCanvas.height = 480;
  const captureCtx = captureCanvas.getContext("2d");

  let appMode = cfg.hasPhraseModel ? "phrase" : "alphabet"; // "phrase" | "alphabet"
  let activeLang = "en"; // "en" | "fil"
  let stream = null;
  let running = false;
  let inFlight = false;
  let loopGen = 0;
  let loopTimer = null;
  let activeController = null;
  let consecutiveErrors = 0;
  let confirmFlashTimer = null;
  let manualCapturePending = false;
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let currentRefIdx = 0;

  let lastStatusKey = "";
  let lastQuickPickKey = "";
  let lastComposerKey = "";
  let lastAutocompleteKey = "";
  let lastDwellBucket = -1;

  const PHRASE_INTERVAL_MS = 90;   // ~11 FPS (matches FSL-105 training sampling rate)
  const ALPHABET_INTERVAL_MS = 70; // ~14 FPS

  let latestComposer = { tokens: [], text_buffer: "", formatted_en: "", formatted_fil: "" };
  let catalogData = { signs: [], categories: [], test_samples: [] };

  function speakText(text) {
    const txt = (text || "").trim();
    if (!txt || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(txt);
    utter.lang = activeLang === "fil" ? "fil-PH" : "en-US";
    utter.rate = 0.96;
    window.speechSynthesis.speak(utter);
  }

  function triggerConfirmFlash(durationMs = 320) {
    if (confirmFlashTimer) clearTimeout(confirmFlashTimer);
    detectedBox.classList.add("confirmed-flash");
    confirmFlashTimer = setTimeout(() => {
      detectedBox.classList.remove("confirmed-flash");
      confirmFlashTimer = null;
    }, durationMs);
  }

  function setStatus(text, cls) {
    const key = `${cls}|${text}`;
    if (key === lastStatusKey) return;
    lastStatusKey = key;
    statusBanner.textContent = text;
    statusBanner.className = `status-pill ${cls}`;
  }

  function updateDwellRing(progress) {
    const bucket = Math.round(Math.min(1, Math.max(0, progress)) * 50);
    if (bucket === lastDwellBucket) return;
    lastDwellBucket = bucket;
    if (progress > 0.02) {
      dwellRingHud.classList.add("visible");
      const offset = RING_CIRCUMFERENCE * (1 - Math.min(1, progress));
      dwellRingProgress.style.strokeDashoffset = `${offset}`;
    } else {
      dwellRingHud.classList.remove("visible");
      dwellRingProgress.style.strokeDashoffset = `${RING_CIRCUMFERENCE}`;
    }
  }

  function renderTop3(top3, activeThreshold) {
    const rows = top3List.querySelectorAll(".top3-row");
    const qpLabels = [];
    for (let i = 0; i < 3; i++) {
      const item = top3[i];
      const charEl = rows[i].querySelector(".top3-char");
      const fillEl = rows[i].querySelector(".bar-fill");
      const pctEl = rows[i].querySelector(".top3-pct");
      const threshEl = rows[i].querySelector(".threshold-line");
      threshEl.style.left = `${activeThreshold}%`;

      if (item) {
        const pct = Math.round(item.prob * 100);
        if (charEl.textContent !== item.label) {
          charEl.textContent = item.label;
          charEl.title = item.label;
        }
        fillEl.style.width = `${pct}%`;
        pctEl.textContent = `${pct}%`;
        if (appMode === "phrase") qpLabels.push(item.label);
      } else {
        if (charEl.textContent !== "—") charEl.textContent = "—";
        fillEl.style.width = "0%";
        pctEl.textContent = "0%";
      }
    }

    const qpKey = `${appMode}|${qpLabels.join(",")}`;
    if (qpKey !== lastQuickPickKey) {
      lastQuickPickKey = qpKey;
      quickPickRow.innerHTML = "";
      if (appMode === "phrase") {
        top3.slice(0, 3).forEach((item) => {
          if (!item) return;
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "quick-pick-btn";
          btn.textContent = `+ ${item.label}`;
          btn.title = `Add "${item.label}" to message`;
          btn.addEventListener("click", () => pickPhraseToken(item));
          quickPickRow.appendChild(btn);
        });
      }
    }
  }

  function drawHandPoints(pts, color) {
    if (!pts || pts.length !== 21) return;
    const w = overlay.width;
    const h = overlay.height;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.strokeStyle = color;
    for (const [a, b] of cfg.handConnections) {
      ctx.beginPath();
      ctx.moveTo(pts[a][0] * w, pts[a][1] * h);
      ctx.lineTo(pts[b][0] * w, pts[b][1] * h);
      ctx.stroke();
    }
    pts.forEach((pt, idx) => {
      ctx.beginPath();
      ctx.arc(pt[0] * w, pt[1] * h, idx === 0 ? 4.5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = idx === 0 ? color : "#ffffff";
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = color;
      ctx.stroke();
    });
  }

  function drawHolisticOverlay(data) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!skeletonToggle.checked) return;
    const w = overlay.width;
    const h = overlay.height;
    if (data.pose && data.pose.length === 15) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(251, 191, 36, 0.85)";
      for (const [a, b] of cfg.upperBodyConnections) {
        ctx.beginPath();
        ctx.moveTo(data.pose[a][0] * w, data.pose[a][1] * h);
        ctx.lineTo(data.pose[b][0] * w, data.pose[b][1] * h);
        ctx.stroke();
      }
    }
    drawHandPoints(data.left_hand, "#38bdf8");
    drawHandPoints(data.right_hand, "#34d399");
  }

  function renderComposer(data) {
    if (data.tokens !== undefined) latestComposer.tokens = data.tokens;
    if (data.text_buffer !== undefined) latestComposer.text_buffer = data.text_buffer;
    if (data.formatted_en !== undefined) latestComposer.formatted_en = data.formatted_en;
    if (data.formatted_fil !== undefined) latestComposer.formatted_fil = data.formatted_fil;

    const previewTxt = activeLang === "fil" ? latestComposer.formatted_fil : latestComposer.formatted_en;
    const tokLabels = (latestComposer.tokens || []).map((t) => t.label).join(",");
    const compKey = `${appMode}|${activeLang}|${tokLabels}|${latestComposer.text_buffer}|${previewTxt}`;
    if (compKey === lastComposerKey) return;
    lastComposerKey = compKey;

    composerTokens.innerHTML = "";
    if (appMode === "phrase" || latestComposer.tokens.length > 0) {
      latestComposer.tokens.forEach((tok) => {
        const chip = document.createElement("span");
        chip.className = "composer-token-chip";
        chip.textContent = tok.label;
        composerTokens.appendChild(chip);
      });
    } else if (latestComposer.text_buffer) {
      const chip = document.createElement("span");
      chip.className = "composer-token-chip";
      chip.textContent = `SPELLED: ${latestComposer.text_buffer}`;
      composerTokens.appendChild(chip);
    }

    if (previewTxt && previewTxt.trim()) {
      naturalPreview.textContent = previewTxt;
    } else {
      naturalPreview.innerHTML = '<span class="text-placeholder-inline">Sign phrases or spell words to compose a natural message...</span>';
    }
  }

  function renderAutocomplete(suggestions) {
    const items = (appMode === "alphabet" && suggestions) ? suggestions : [];
    const acKey = `${appMode}|${items.join(",")}`;
    if (acKey === lastAutocompleteKey) return;
    lastAutocompleteKey = acKey;

    autocompleteRow.innerHTML = "";
    if (items.length === 0) {
      autocompleteRow.classList.add("hidden");
      return;
    }
    autocompleteRow.classList.remove("hidden");
    items.forEach((word) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "autocomplete-chip";
      btn.textContent = word;
      btn.addEventListener("click", async () => {
        const res = await fetch("/api/composer/complete_word", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ word }),
        });
        const data = await res.json();
        renderComposer(data);
        renderAutocomplete([]);
      });
      autocompleteRow.appendChild(btn);
    });
  }

  function syncModelOptions() {
    modelSelect.innerHTML = "";
    if (appMode === "phrase") {
      modelSelect.add(new Option("FSLPhraseNet (Main)", "FSLPhraseNet"));
      if (cfg.hasPhraseSvm) modelSelect.add(new Option("FSL-105 SVM (Baseline)", "FSL-105 SVM"));
    } else {
      modelSelect.add(new Option("Deep MLP (Main)", "Deep MLP"));
      if (cfg.hasSvm) modelSelect.add(new Option("Baseline SVM", "Baseline SVM"));
    }
  }

  function setMode(nextMode) {
    appMode = nextMode;
    modePhraseBtn.classList.toggle("is-active", appMode === "phrase");
    modeSpellBtn.classList.toggle("is-active", appMode === "alphabet");

    const isPhrase = appMode === "phrase";
    modeTag.textContent = isPhrase ? "FSL-105 PHRASE" : "A–Z SPELL";
    detectedSub.textContent = isPhrase ? "Raise hands to sign" : "Hold sign 1.0s";
    detectedLetter.textContent = "—";
    const thresh = isPhrase ? cfg.phraseThreshold : cfg.confidenceThreshold;
    thresholdTag.innerHTML = `&tau; = ${thresh}%`;

    captureNowBtn.classList.toggle("hidden", !isPhrase);
    spaceBtn.classList.toggle("hidden", isPhrase);

    syncModelOptions();
    lastQuickPickKey = "";
    lastComposerKey = "";
    lastAutocompleteKey = "";
    renderTop3([], thresh);
    renderAutocomplete([]);
    renderComposer({});
    updateDwellRing(0);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    fetch("/api/composer/reset_phrase_capture", { method: "POST" }).catch(() => {});
  }

  async function pickPhraseToken(item) {
    const res = await fetch("/api/composer/pick_phrase", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(item),
    });
    const data = await res.json();
    renderComposer(data);
    triggerConfirmFlash(300);
  }

  function speakComposerOutput() {
    const txt = activeLang === "fil" ? latestComposer.formatted_fil : latestComposer.formatted_en;
    speakText(txt);
  }

  function scheduleNextTick(gen, delayMs) {
    if (!running || gen !== loopGen) return;
    if (loopTimer) clearTimeout(loopTimer);
    loopTimer = setTimeout(() => {
      loopTimer = null;
      processLoop(gen);
    }, delayMs);
  }

  async function processLoop(gen) {
    if (!running || gen !== loopGen) return;
    if (document.hidden || video.readyState < 2 || inFlight) {
      scheduleNextTick(gen, document.hidden ? 300 : 80);
      return;
    }

    const tickStart = performance.now();
    const targetInterval = appMode === "phrase" ? PHRASE_INTERVAL_MS : ALPHABET_INTERVAL_MS;
    inFlight = true;

    const controller = new AbortController();
    activeController = controller;
    const reqTimeout = setTimeout(() => controller.abort(), 4000);

    try {
      const capW = appMode === "phrase" ? 480 : 640;
      const capH = appMode === "phrase" ? 360 : 480;
      if (captureCanvas.width !== capW || captureCanvas.height !== capH) {
        captureCanvas.width = capW;
        captureCanvas.height = capH;
      }

      captureCtx.save();
      if (appMode === "alphabet" && mirrorToggle.checked) {
        captureCtx.translate(capW, 0);
        captureCtx.scale(-1, 1);
      }
      captureCtx.drawImage(video, 0, 0, capW, capH);
      captureCtx.restore();

      const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.68);

      if (appMode === "phrase") {
        const triggerManual = manualCapturePending;
        manualCapturePending = false;
        const res = await fetch("/api/predict_phrase_frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            image: dataUrl,
            model: modelSelect.value,
            clahe: claheToggle.checked,
            auto_clahe: autoClaheToggle.checked,
            mirrored: mirrorToggle.checked,
            lang: activeLang,
            manual_capture: triggerManual,
          }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!running || gen !== loopGen) return;

        consecutiveErrors = 0;
        claheBadge.classList.toggle("hidden", !data.clahe_assisted);
        if (data.capture_state === "signing") {
          setStatus(`Signing (${data.active_frames}f)`, "status-ok");
        } else if (data.has_hands) {
          setStatus("Hands Ready", "status-ok");
        } else {
          setStatus("Raise hands to sign", "status-warn");
        }

        drawHolisticOverlay(data);
        renderTop3(data.top3 || [], cfg.phraseThreshold);
        updateDwellRing(data.progress || 0);

        if (data.top3 && data.top3.length > 0) {
          const top = data.top3[0];
          if (detectedLetter.textContent !== top.label) detectedLetter.textContent = top.label;
          detectedSub.textContent = `${Math.round(top.prob * 100)}% • ${top.category}`;
          detectedBox.classList.toggle("above-threshold", top.prob * 100 >= cfg.phraseThreshold);
        } else {
          if (detectedLetter.textContent !== "—") detectedLetter.textContent = "—";
          detectedSub.textContent = data.capture_state === "signing" ? "Collecting motion..." : "Raise hands to sign";
          detectedBox.classList.remove("above-threshold");
        }

        if (data.just_confirmed) triggerConfirmFlash(350);
        renderComposer(data);
        latencyHud.textContent = `Latency: ${data.latency_ms} ms`;
      } else {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            image: dataUrl,
            model: modelSelect.value,
            clahe: claheToggle.checked,
            auto_clahe: autoClaheToggle.checked,
            auto_space: autoSpaceToggle.checked,
          }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!running || gen !== loopGen) return;

        consecutiveErrors = 0;
        claheBadge.classList.toggle("hidden", !data.clahe_assisted);
        if (data.hand_detected) setStatus("Hand Tracked", "status-ok");
        else setStatus("No hand detected", "status-warn");

        ctx.clearRect(0, 0, overlay.width, overlay.height);
        if (skeletonToggle.checked) drawHandPoints(data.landmarks, "#38bdf8");
        renderTop3(data.top3 || [], cfg.confidenceThreshold);

        const activeChar = data.smoothed_label || data.raw_label || "—";
        const aboveThreshold = (data.raw_confidence || 0) * 100 >= cfg.confidenceThreshold;
        if (detectedLetter.textContent !== activeChar) detectedLetter.textContent = activeChar;
        detectedSub.textContent = data.hand_detected ? `${Math.round((data.raw_confidence || 0) * 100)}% conf` : "Hold sign 1.0s";
        detectedBox.classList.toggle("above-threshold", Boolean(data.hand_detected && aboveThreshold));

        if (data.just_confirmed) triggerConfirmFlash(350);
        updateDwellRing(data.dwell_progress || 0);
        renderComposer(data);
        renderAutocomplete(data.suggestions || []);
        latencyHud.textContent = `Latency: ${data.latency_ms} ms`;
      }

      frameCount++;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        fpsHud.textContent = `FPS: ${frameCount}`;
        frameCount = 0;
        lastFpsTime = now;
      }
    } catch (err) {
      if (err && err.name !== "AbortError") {
        consecutiveErrors++;
        console.warn("Inference request error:", err);
        setStatus("Reconnecting to server...", "status-warn");
      }
    } finally {
      clearTimeout(reqTimeout);
      if (activeController === controller) activeController = null;
      inFlight = false;
    }

    if (!running || gen !== loopGen) return;
    const elapsed = performance.now() - tickStart;
    const backoff = consecutiveErrors > 0 ? Math.min(3000, 600 * Math.pow(2, consecutiveErrors - 1)) : 0;
    const nextDelay = Math.max(16, targetInterval - elapsed, backoff);
    scheduleNextTick(gen, nextDelay);
  }

  async function populateCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const vids = devices.filter((d) => d.kind === "videoinput");
      if (!vids.length) return;
      const prev = cameraSelect.value;
      cameraSelect.innerHTML = "";
      vids.forEach((dev, i) => {
        const opt = document.createElement("option");
        opt.value = dev.deviceId;
        opt.textContent = dev.label || `Camera ${i + 1}`;
        cameraSelect.appendChild(opt);
      });
      if (prev) cameraSelect.value = prev;
    } catch {}
  }

  async function startCamera() {
    if (running) stopCamera();
    try {
      const vc = { width: 640, height: 480 };
      if (cameraSelect.value) vc.deviceId = { exact: cameraSelect.value };
      stream = await navigator.mediaDevices.getUserMedia({ video: vc, audio: false });
      await populateCameras();
      video.srcObject = stream;
      video.style.transform = mirrorToggle.checked ? "scaleX(-1)" : "none";
      placeholder.style.display = "none";
      running = true;
      consecutiveErrors = 0;
      frameCount = 0;
      lastFpsTime = performance.now();
      loopGen += 1;
      const myGen = loopGen;
      toggleCamBtn.textContent = "Stop Camera";
      toggleCamBtn.classList.add("is-recording");
      setStatus("Camera Active", "status-ok");
      scheduleNextTick(myGen, 60);
    } catch (err) {
      setStatus("Camera permission denied or unavailable", "status-warn");
      alert("Could not access webcam: " + err.message);
    }
  }

  function stopCamera() {
    running = false;
    loopGen += 1;
    if (loopTimer) {
      clearTimeout(loopTimer);
      loopTimer = null;
    }
    if (activeController) {
      activeController.abort();
      activeController = null;
    }
    inFlight = false;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.srcObject = null;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    updateDwellRing(0);
    claheBadge.classList.add("hidden");
    placeholder.style.display = "flex";
    toggleCamBtn.textContent = "Start Camera";
    toggleCamBtn.classList.remove("is-recording");
    setStatus("Camera Offline", "status-idle");
    latencyHud.textContent = "Latency: -- ms";
    fpsHud.textContent = "FPS: --";
    fetch("/api/composer/reset_phrase_capture", { method: "POST" }).catch(() => {});
  }

  async function loadCatalogAndChat() {
    try {
      const [catRes, convRes] = await Promise.all([fetch("/api/fsl105/catalog"), fetch("/api/conversation")]);
      if (catRes.ok) {
        catalogData = await catRes.json();
        populateSignGuide();
      }
      if (convRes.ok) {
        const cdata = await convRes.json();
        renderComposer(cdata);
      }
    } catch (err) {
      console.warn("Catalog/chat init error:", err);
    }
  }

  function populateSignGuide() {
    guideCategorySelect.innerHTML = `<option value="ALL">All Categories (${catalogData.signs.length})</option>`;
    (catalogData.categories || []).forEach((cat) => {
      guideCategorySelect.add(new Option(cat, cat));
    });
    renderGuideGrid();
  }

  function renderGuideGrid() {
    const q = (guideSearchInput.value || "").trim().toLowerCase();
    const cat = guideCategorySelect.value || "ALL";
    guidePhraseGrid.innerHTML = "";

    const filtered = (catalogData.signs || []).filter((s) => {
      const matchCat = cat === "ALL" || s.category === cat;
      const matchQ = !q || s.label.toLowerCase().includes(q) || s.en.toLowerCase().includes(q) || s.fil.toLowerCase().includes(q);
      return matchCat && matchQ;
    });

    filtered.forEach((s) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "guide-sign-card";
      card.dataset.signId = String(s.id);
      card.innerHTML = `
        <img src="${s.thumb_url}" alt="${s.label}" loading="lazy" />
        <div class="guide-sign-card-body">
          <div class="guide-sign-label">${s.label}</div>
          <div class="guide-sign-cat">${s.category}</div>
        </div>
      `;
      card.addEventListener("click", () => showGuideSignDetail(s));
      guidePhraseGrid.appendChild(card);
    });
  }

  function showGuideSignDetail(sign) {
    document.querySelectorAll(".guide-sign-card.is-selected").forEach((el) => el.classList.remove("is-selected"));
    const el = guidePhraseGrid.querySelector(`[data-sign-id="${sign.id}"]`);
    if (el) el.classList.add("is-selected");

    guideDetailCard.innerHTML = `
      <video class="guide-detail-video" src="${sign.video_url}" controls autoplay loop muted playsinline></video>
      <div><span class="mode-badge">${sign.category}</span></div>
      <div class="fw-bold fs-6">${sign.label}</div>
      <div class="small"><strong>English:</strong> ${sign.en}</div>
      <div class="small"><strong>Filipino:</strong> ${sign.fil}</div>
      <button type="button" id="guideAddSignBtn" class="action-pill-btn action-pill-primary mt-1">Add to Message</button>
    `;
    document.getElementById("guideAddSignBtn").addEventListener("click", async () => {
      await pickPhraseToken(sign);
      signGuideModal.classList.add("hidden");
    });
  }

  function openSignGuideAt(signId) {
    signGuideModal.classList.remove("hidden");
    guideTabPhrases.classList.add("is-active");
    guideTabAlphabet.classList.remove("is-active");
    guidePhrasesPane.classList.remove("hidden");
    guideAlphabetPane.classList.add("hidden");

    const found = (catalogData.signs || []).find((s) => s.id === signId);
    if (found) showGuideSignDetail(found);
  }

  // Event Wiring
  modePhraseBtn.addEventListener("click", () => setMode("phrase"));
  modeSpellBtn.addEventListener("click", () => setMode("alphabet"));
  toggleCamBtn.addEventListener("click", () => (running ? stopCamera() : startCamera()));
  captureNowBtn.addEventListener("click", () => { if (running) manualCapturePending = true; });

  langToggleBtn.addEventListener("click", () => {
    activeLang = activeLang === "en" ? "fil" : "en";
    langBadgeText.textContent = activeLang.toUpperCase();
    composerHint.textContent = activeLang === "fil" ? "Natural FIL output" : "Natural EN output";
    renderComposer({});
  });

  stageMirrorBtn.addEventListener("click", () => {
    mirrorToggle.checked = !mirrorToggle.checked;
    stageMirrorBtn.classList.toggle("is-active", mirrorToggle.checked);
    stageMirrorBtn.title = mirrorToggle.checked ? "Mirror View (On)" : "Mirror View (Off)";
    video.style.transform = mirrorToggle.checked ? "scaleX(-1)" : "none";
  });

  settingsGearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isHidden = settingsPopover.classList.toggle("hidden");
    settingsGearBtn.setAttribute("aria-expanded", String(!isHidden));
  });
  document.addEventListener("click", (e) => {
    if (!settingsPopover.classList.contains("hidden") && !settingsPopover.contains(e.target) && !settingsGearBtn.contains(e.target)) {
      settingsPopover.classList.add("hidden");
    }
  });

  sendSpeakBtn.addEventListener("click", speakComposerOutput);
  spaceBtn.addEventListener("click", async () => {
    const res = await fetch("/api/buffer/space", { method: "POST" });
    renderComposer(await res.json());
  });
  undoBtn.addEventListener("click", async () => {
    const res = await fetch("/api/composer/undo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: appMode }),
    });
    renderComposer(await res.json());
  });
  clearComposerBtn.addEventListener("click", async () => {
    const res = await fetch("/api/composer/clear", { method: "POST" });
    renderComposer(await res.json());
    renderAutocomplete([]);
  });

  // Sign Guide & A-Z Reference Modal
  openSignGuideBtn.addEventListener("click", () => signGuideModal.classList.remove("hidden"));
  signGuideClose.addEventListener("click", () => signGuideModal.classList.add("hidden"));
  signGuideBackdrop.addEventListener("click", () => signGuideModal.classList.add("hidden"));
  guideTabPhrases.addEventListener("click", () => {
    guideTabPhrases.classList.add("is-active");
    guideTabAlphabet.classList.remove("is-active");
    guidePhrasesPane.classList.remove("hidden");
    guideAlphabetPane.classList.add("hidden");
  });
  guideTabAlphabet.addEventListener("click", () => {
    guideTabAlphabet.classList.add("is-active");
    guideTabPhrases.classList.remove("is-active");
    guideAlphabetPane.classList.remove("hidden");
    guidePhrasesPane.classList.add("hidden");
  });
  guideSearchInput.addEventListener("input", renderGuideGrid);
  guideCategorySelect.addEventListener("change", renderGuideGrid);

  function selectRefSample(letter, idx) {
    refModalImg.src = `/api/reference/${letter}?idx=${idx}`;
    refSampleThumbs.forEach((b, i) => b.classList.toggle("active", i === idx));
  }
  function openRefModal(index) {
    currentRefIdx = (index + cheatTiles.length) % cheatTiles.length;
    const letter = cheatTiles[currentRefIdx].dataset.letter;
    refModalBadge.textContent = letter;
    refModalTitle.textContent = `FSL Alphabet Reference — "${letter}"`;
    refSampleThumbs.forEach((b, i) => { b.querySelector("img").src = `/api/reference/${letter}?idx=${i}`; });
    selectRefSample(letter, 0);
    refModalCounter.textContent = `${currentRefIdx + 1} / ${cheatTiles.length}`;
    refModal.classList.remove("hidden");
  }
  cheatTiles.forEach((tile, idx) => tile.addEventListener("click", () => openRefModal(idx)));
  refSampleThumbs.forEach((b, i) => b.addEventListener("click", () => selectRefSample(cheatTiles[currentRefIdx].dataset.letter, i)));
  refModalClose.addEventListener("click", () => refModal.classList.add("hidden"));
  refModalBackdrop.addEventListener("click", () => refModal.classList.add("hidden"));
  refPrevBtn.addEventListener("click", () => openRefModal(currentRefIdx - 1));
  refNextBtn.addEventListener("click", () => openRefModal(currentRefIdx + 1));

  cameraSelect.addEventListener("change", () => {
    if (running) { stopCamera(); startCamera(); }
  });
  skeletonToggle.addEventListener("change", () => {
    if (!skeletonToggle.checked) ctx.clearRect(0, 0, overlay.width, overlay.height);
  });

  setMode(appMode);
  populateCameras();
  loadCatalogAndChat();
})();
