/**
 * WII-AI Windshield Studio — Frontend Application
 * Handles ROI polygon editing, frame sampling estimation, live annotated streaming,
 * coordinates inspection with pixel/YOLO bounds, and dataset exports.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    videoMeta: null,
    currentVideoPath: null,
    roiPoints: [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0]
    ],
    roiDraggingIndex: -1,
    roiReferenceFrame: 0,
    roiImage: null,
    isProcessing: false,
    telemetryInterval: null,
    windshieldRecords: [],
    recordsByFrame: {},
    uniqueFramesWithDetections: [],
    inspectorCurrentFrame: 0,
    selectedWindshieldId: null,
    inspectorImage: null
  };

  // DOM Elements
  const videoDropdown = document.getElementById('video-dropdown');
  const btnBrowseFile = document.getElementById('btn-browse-file');
  const videoFileInput = document.getElementById('video-file-input');
  const roiStatusBadge = document.getElementById('roi-status-badge');
  const badgeVideoStatus = document.getElementById('badge-video-status');
  const badgeWindshieldsCount = document.getElementById('badge-windshields-count');

  // Metadata elements
  const metaResolution = document.getElementById('meta-resolution');
  const metaAspect = document.getElementById('meta-aspect');
  const metaFps = document.getElementById('meta-fps');
  const metaDuration = document.getElementById('meta-duration');
  const metaFrames = document.getElementById('meta-frames');
  const metaFilesize = document.getElementById('meta-filesize');
  const metaFormat = document.getElementById('meta-format');

  // Activity Banner
  const activityCard = document.getElementById('activity-card');
  const activityTitle = document.getElementById('activity-title');
  const activityDetail = document.getElementById('activity-detail');
  const activityState = document.getElementById('activity-state');
  const activityProgressBar = document.getElementById('activity-progress-bar');

  // Tabs
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  // Tab 1: ROI
  const roiCanvas = document.getElementById('roi-canvas');
  const roiCtx = roiCanvas.getContext('2d');
  const roiFrameSlider = document.getElementById('roi-frame-slider');
  const roiFrameLabel = document.getElementById('roi-frame-label');
  const roiVertexCount = document.getElementById('roi-vertex-count');
  const roiCoordsPreview = document.getElementById('roi-coords-preview');
  const paramRoiMatch = document.getElementById('param-roi-match');
  const btnUndoPoint = document.getElementById('btn-undo-point');
  const btnClearRoi = document.getElementById('btn-clear-roi');
  const btnResetRoi = document.getElementById('btn-reset-roi');
  const btnConfirmRoi = document.getElementById('btn-confirm-roi');
  const btnPresetRoad = document.getElementById('btn-preset-road');
  const btnPresetLeft = document.getElementById('btn-preset-left');
  const btnPresetRight = document.getElementById('btn-preset-right');
  const btnPresetFull = document.getElementById('btn-preset-full');

  // Tab 2: Frame Selection & Sampling
  const paramStartFrame = document.getElementById('param-start-frame');
  const paramEndFrame = document.getElementById('param-end-frame');
  const paramFrameStep = document.getElementById('param-frame-step');
  const paramWindshieldConf = document.getElementById('param-windshield-conf');
  const labelWindshieldConf = document.getElementById('label-windshield-conf');
  const chkSaveFrames = document.getElementById('chk-save-frames');
  const estTotalFrames = document.getElementById('est-total-frames');
  const estDuration = document.getElementById('est-duration');
  const estActiveArea = document.getElementById('est-active-area');
  const btnStartExtract = document.getElementById('btn-start-extract');

  // Tab 3: Live Extraction Stream
  const liveFrameImg = document.getElementById('live-frame-img');
  const livePlaceholderBox = document.getElementById('live-placeholder-box');
  const liveStreamBadge = document.getElementById('live-stream-badge');
  const telemetryStatusPill = document.getElementById('telemetry-status-pill');
  const statWindshieldsFound = document.getElementById('stat-windshields-found');
  const statFramesDone = document.getElementById('stat-frames-done');
  const statFramesTotal = document.getElementById('stat-frames-total');
  const statFps = document.getElementById('stat-fps');
  const statEta = document.getElementById('stat-eta');
  const liveConsoleLog = document.getElementById('live-console-log');
  const btnPauseExtract = document.getElementById('btn-pause-extract');
  const btnResumeExtract = document.getElementById('btn-resume-extract');
  const btnStopExtract = document.getElementById('btn-stop-extract');
  const btnGotoInspector = document.getElementById('btn-goto-inspector');

  // Tab 4: Inspector
  const inspectorCanvas = document.getElementById('inspector-canvas');
  const inspectorCtx = inspectorCanvas.getContext('2d');
  const inspectorFrameBadge = document.getElementById('inspector-frame-badge');
  const btnPrevWindshieldFrame = document.getElementById('btn-prev-windshield-frame');
  const btnNextWindshieldFrame = document.getElementById('btn-next-windshield-frame');
  const inspectorYoloPreview = document.getElementById('inspector-yolo-preview');
  const inspectorTotalCount = document.getElementById('inspector-total-count');
  const filterWindshieldText = document.getElementById('filter-windshield-text');
  const windshieldsContainer = document.getElementById('windshields-container');
  const btnRefreshWindshields = document.getElementById('btn-refresh-windshields');
  const btnClearWindshields = document.getElementById('btn-clear-windshields');

  // Tab 5: Export
  const exportSplitSlider = document.getElementById('export-split-slider');
  const exportSplitLabel = document.getElementById('export-split-label');
  const btnDownloadYoloZip = document.getElementById('btn-download-yolo-zip');

  // --- TAB NAVIGATION ---
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      switchTab(targetId);
    });
  });

  function switchTab(tabId) {
    tabButtons.forEach(b => b.classList.toggle('active', b.getAttribute('data-tab') === tabId));
    tabContents.forEach(c => c.classList.toggle('active', c.id === tabId));

    if (tabId === 'tab-roi') {
      redrawRoiCanvas();
    } else if (tabId === 'tab-source') {
      updateEstimation();
    } else if (tabId === 'tab-inspector') {
      loadInspectorResults();
    }
  }

  // --- VIDEO SOURCE & INITIALIZATION ---
  async function loadVideoList() {
    try {
      const res = await fetch('/api/video/list');
      const data = await res.json();
      videoDropdown.innerHTML = '';

      if (data.videos && data.videos.length > 0) {
        data.videos.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v.path;
          opt.textContent = `${v.display} (${v.size_mb} MB)`;
          if (v.path === data.current_video) opt.selected = true;
          videoDropdown.appendChild(opt);
        });
        loadVideoInfo(videoDropdown.value);
      } else {
        videoDropdown.innerHTML = '<option value="">No videos found - browse to upload</option>';
        badgeVideoStatus.textContent = 'No Video Loaded';
      }
    } catch (err) {
      console.error('Error loading video list:', err);
    }
  }

  videoDropdown.addEventListener('change', () => {
    if (videoDropdown.value) {
      loadVideoInfo(videoDropdown.value);
    }
  });

  btnBrowseFile.addEventListener('click', () => videoFileInput.click());

  videoFileInput.addEventListener('change', async () => {
    if (!videoFileInput.files || videoFileInput.files.length === 0) return;
    const file = videoFileInput.files[0];
    const formData = new FormData();
    formData.append('video', file);

    setActivity('Uploading Video', `Uploading ${file.name}...`, 'PROCESSING', 40);
    badgeVideoStatus.textContent = 'Uploading...';

    try {
      const res = await fetch('/api/video/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok) {
        setActivity('Studio Ready', `Loaded video: ${file.name}`, 'IDLE', 0);
        await loadVideoList();
      } else {
        alert(data.error || 'Failed to upload video');
        setActivity('Error', 'Video upload failed', 'IDLE', 0);
      }
    } catch (err) {
      console.error('Upload error:', err);
      alert('Upload failed: ' + err.message);
    }
  });

  async function loadVideoInfo(videoPath) {
    badgeVideoStatus.textContent = 'Loading Metadata...';
    try {
      const res = await fetch(`/api/video/info?path=${encodeURIComponent(videoPath)}`);
      const meta = await res.json();
      if (!res.ok) throw new Error(meta.error || 'Failed to load video info');

      state.videoMeta = meta;
      state.currentVideoPath = meta.filepath;

      metaResolution.textContent = meta.resolution || '-- x --';
      metaAspect.textContent = `Aspect: ${meta.aspect_ratio || '--'}`;
      metaFps.textContent = `${meta.fps || 0} fps`;
      metaDuration.textContent = meta.duration_formatted || '--:--:--';
      metaFrames.textContent = `${meta.total_frames || 0} frames`;
      metaFilesize.textContent = meta.filesize_formatted || '-- MB';
      metaFormat.textContent = `Format: ${meta.format || '--'}`;

      badgeVideoStatus.textContent = `Ready · ${meta.filename}`;

      // Update frame scrubber ranges
      const totalF = meta.total_frames || 1000;
      roiFrameSlider.max = Math.max(0, totalF - 1);
      roiFrameSlider.value = 0;
      paramStartFrame.value = 0;
      paramEndFrame.value = Math.min(500, totalF);

      updateScrubberLabel();
      updateEstimation();
      loadRoiFrameImage(0);
    } catch (err) {
      console.error('Video info error:', err);
      badgeVideoStatus.textContent = 'Video Error';
    }
  }

  // --- ROI CANVAS STUDIO ---
  function updateScrubberLabel() {
    const cur = roiFrameSlider.value;
    const maxF = roiFrameSlider.max;
    roiFrameLabel.textContent = `Frame ${cur} / ${maxF}`;
  }

  roiFrameSlider.addEventListener('input', () => {
    updateScrubberLabel();
    loadRoiFrameImage(parseInt(roiFrameSlider.value, 10));
  });

  function loadRoiFrameImage(frameIdx) {
    if (!state.currentVideoPath) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = `/api/video/frame?path=${encodeURIComponent(state.currentVideoPath)}&frame_idx=${frameIdx}&max_dim=960&t=${Date.now()}`;
    img.onload = () => {
      state.roiImage = img;
      redrawRoiCanvas();
    };
  }

  function redrawRoiCanvas() {
    if (!state.roiImage) return;

    // Set canvas resolution to image aspect ratio
    const container = document.getElementById('roi-canvas-wrapper');
    const cw = container.clientWidth || 640;
    const aspect = state.roiImage.naturalHeight / state.roiImage.naturalWidth;
    roiCanvas.width = cw;
    roiCanvas.height = Math.round(cw * aspect);

    const w = roiCanvas.width;
    const h = roiCanvas.height;

    // Draw background video frame
    roiCtx.drawImage(state.roiImage, 0, 0, w, h);

    // Draw ROI Polygon
    if (state.roiPoints.length > 0) {
      roiCtx.beginPath();
      state.roiPoints.forEach((pt, i) => {
        const px = pt[0] * w;
        const py = pt[1] * h;
        if (i === 0) roiCtx.moveTo(px, py);
        else roiCtx.lineTo(px, py);
      });
      roiCtx.closePath();

      // Neon Cyan Translucent fill
      roiCtx.fillStyle = 'rgba(0, 229, 255, 0.20)';
      roiCtx.fill();

      // Neon Cyan Stroke
      roiCtx.strokeStyle = '#00e5ff';
      roiCtx.lineWidth = 2.5;
      roiCtx.stroke();

      // Draw draggable handles
      state.roiPoints.forEach((pt, i) => {
        const px = pt[0] * w;
        const py = pt[1] * h;

        roiCtx.beginPath();
        roiCtx.arc(px, py, 7, 0, Math.PI * 2);
        roiCtx.fillStyle = '#ffffff';
        roiCtx.fill();
        roiCtx.strokeStyle = '#00e5ff';
        roiCtx.lineWidth = 2.5;
        roiCtx.stroke();

        // Vertex index label
        roiCtx.fillStyle = '#0f172a';
        roiCtx.font = 'bold 9px monospace';
        roiCtx.textAlign = 'center';
        roiCtx.textBaseline = 'middle';
        roiCtx.fillText(i + 1, px, py);
      });
    }

    // Update status previews
    updateRoiPreviewText();
  }

  function updateRoiPreviewText() {
    const n = state.roiPoints.length;
    roiVertexCount.textContent = `${n} Vertices (${n >= 3 ? 'Active Polygon' : 'Incomplete'})`;
    const formatted = state.roiPoints.map(p => `[${p[0].toFixed(3)}, ${p[1].toFixed(3)}]`);
    roiCoordsPreview.textContent = '[\n  ' + formatted.join(',\n  ') + '\n]';

    const isFull = state.roiPoints.length === 4 &&
      state.roiPoints[0][0] === 0 && state.roiPoints[0][1] === 0 &&
      state.roiPoints[1][0] === 1 && state.roiPoints[1][1] === 0 &&
      state.roiPoints[2][0] === 1 && state.roiPoints[2][1] === 1 &&
      state.roiPoints[3][0] === 0 && state.roiPoints[3][1] === 1;

    roiStatusBadge.textContent = isFull ? 'ROI: Full Frame' : `ROI: ${n} Points Active`;
    roiStatusBadge.className = isFull ? 'badge' : 'badge warning';
  }

  // Canvas Mouse Interaction for ROI
  function getCanvasCoords(e) {
    const rect = roiCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    return [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))];
  }

  roiCanvas.addEventListener('mousedown', e => {
    const [normX, normY] = getCanvasCoords(e);
    const w = roiCanvas.width;
    const h = roiCanvas.height;

    // Check if clicked near an existing vertex (threshold ~14px)
    let foundIndex = -1;
    for (let i = 0; i < state.roiPoints.length; i++) {
      const [vx, vy] = state.roiPoints[i];
      const dist = Math.hypot((vx - normX) * w, (vy - normY) * h);
      if (dist <= 14) {
        foundIndex = i;
        break;
      }
    }

    if (foundIndex >= 0) {
      state.roiDraggingIndex = foundIndex;
    } else {
      // Add new vertex point
      state.roiPoints.push([Math.round(normX * 1000) / 1000, Math.round(normY * 1000) / 1000]);
      redrawRoiCanvas();
    }
  });

  window.addEventListener('mousemove', e => {
    if (state.roiDraggingIndex >= 0) {
      const [normX, normY] = getCanvasCoords(e);
      state.roiPoints[state.roiDraggingIndex] = [
        Math.round(normX * 1000) / 1000,
        Math.round(normY * 1000) / 1000
      ];
      redrawRoiCanvas();
    }
  });

  window.addEventListener('mouseup', () => {
    state.roiDraggingIndex = -1;
  });

  btnUndoPoint.addEventListener('click', () => {
    if (state.roiPoints.length > 0) {
      state.roiPoints.pop();
      redrawRoiCanvas();
    }
  });

  btnClearRoi.addEventListener('click', () => {
    state.roiPoints = [];
    redrawRoiCanvas();
  });

  btnResetRoi.addEventListener('click', () => {
    setPresetFull();
  });

  btnConfirmRoi.addEventListener('click', () => {
    switchTab('tab-source');
  });

  // Presets
  function setPresetFull() {
    state.roiPoints = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    redrawRoiCanvas();
    updateEstimation();
  }

  function setPresetRoad() {
    state.roiPoints = [[0.05, 0.35], [0.95, 0.35], [0.95, 0.98], [0.05, 0.98]];
    redrawRoiCanvas();
    updateEstimation();
  }

  function setPresetLeft() {
    state.roiPoints = [[0.05, 0.35], [0.48, 0.35], [0.48, 0.98], [0.05, 0.98]];
    redrawRoiCanvas();
    updateEstimation();
  }

  function setPresetRight() {
    state.roiPoints = [[0.52, 0.35], [0.95, 0.35], [0.95, 0.98], [0.52, 0.98]];
    redrawRoiCanvas();
    updateEstimation();
  }

  btnPresetFull.addEventListener('click', setPresetFull);
  btnPresetRoad.addEventListener('click', setPresetRoad);
  btnPresetLeft.addEventListener('click', setPresetLeft);
  btnPresetRight.addEventListener('click', setPresetRight);

  // Resize canvas when window is resized
  window.addEventListener('resize', () => {
    if (document.getElementById('tab-roi').classList.contains('active')) {
      redrawRoiCanvas();
    }
  });

  // --- TAB 2: FRAME SELECTION & ESTIMATION ---
  function computePolygonArea(pts) {
    if (!pts || pts.length < 3) return 1.0;
    let area = 0.0;
    for (let i = 0; i < pts.length; i++) {
      const j = (i + 1) % pts.length;
      area += pts[i][0] * pts[j][1];
      area -= pts[j][0] * pts[i][1];
    }
    return Math.abs(area) / 2.0;
  }

  function updateEstimation() {
    const startF = Math.max(0, parseInt(paramStartFrame.value, 10) || 0);
    const endF = Math.max(startF + 1, parseInt(paramEndFrame.value, 10) || (state.videoMeta?.total_frames || 500));
    const step = Math.max(1, parseInt(paramFrameStep.value, 10) || 5);

    const totalToScan = Math.ceil((endF - startF) / step);
    estTotalFrames.textContent = `${totalToScan} frames`;

    // Estimation: ~10 frames per second on average with local acceleration
    const estSec = Math.max(1, Math.round(totalToScan / 8.0));
    estDuration.textContent = estSec < 60 ? `~${estSec} sec` : `~${Math.round(estSec / 60)} min ${estSec % 60}s`;

    const areaRatio = computePolygonArea(state.roiPoints);
    const pct = Math.round(areaRatio * 100);
    estActiveArea.textContent = pct >= 98 ? 'Full Frame (100% of video)' : `Selected Area (~${pct}% of frame)`;
  }

  [paramStartFrame, paramEndFrame, paramFrameStep].forEach(el => {
    el.addEventListener('input', updateEstimation);
  });

  paramWindshieldConf.addEventListener('input', () => {
    labelWindshieldConf.textContent = `Threshold: ${parseFloat(paramWindshieldConf.value).toFixed(2)} (OWL-ViT)`;
  });

  // Frame Count Quick Buttons
  function applyFrameCountPreset(count) {
    const total = state.videoMeta?.total_frames || 1000;
    paramStartFrame.value = 0;
    if (count === 'full') {
      paramEndFrame.value = total;
    } else {
      paramEndFrame.value = Math.min(total, count);
    }
    updateEstimation();
  }

  document.getElementById('btn-frames-50').addEventListener('click', () => applyFrameCountPreset(50));
  document.getElementById('btn-frames-100').addEventListener('click', () => applyFrameCountPreset(100));
  document.getElementById('btn-frames-300').addEventListener('click', () => applyFrameCountPreset(300));
  document.getElementById('btn-frames-500').addEventListener('click', () => applyFrameCountPreset(500));
  document.getElementById('btn-frames-1000').addEventListener('click', () => applyFrameCountPreset(1000));
  document.getElementById('btn-frames-full').addEventListener('click', () => applyFrameCountPreset('full'));

  // --- TAB 3: EXTRACTION WORKFLOW & STREAM ---
  btnStartExtract.addEventListener('click', async () => {
    if (!state.currentVideoPath) {
      alert('Please load or select a video first.');
      return;
    }

    const startF = Math.max(0, parseInt(paramStartFrame.value, 10) || 0);
    const endF = Math.max(startF + 1, parseInt(paramEndFrame.value, 10) || (state.videoMeta?.total_frames || 500));
    const step = Math.max(1, parseInt(paramFrameStep.value, 10) || 5);
    const wConf = parseFloat(paramWindshieldConf.value) || 0.15;
    const saveFrames = chkSaveFrames.checked;
    const roiMatch = paramRoiMatch.value;

    const payload = {
      video_path: state.currentVideoPath,
      start_frame: startF,
      end_frame: endF,
      frame_step: step,
      windshield_conf: wConf,
      vehicle_conf: 0.30,
      save_frames: saveFrames,
      roi_polygon: state.roiPoints,
      roi_match: roiMatch,
      vehicle_types: ['Car', 'Bus', 'Truck', 'Motorcycle']
    };

    setActivity('Extraction Running', 'Detecting windshields with OWL-ViT zero-shot algorithm...', 'PROCESSING', 5);
    logConsole('[Start] Launching windshield extraction pipeline...');

    try {
      const res = await fetch('/api/video/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (!res.ok) {
        alert(data.error || 'Failed to start extraction');
        setActivity('Ready', 'Studio ready', 'IDLE', 0);
        return;
      }

      // Switch to stream view
      switchTab('tab-extract');
      startLiveStreamMonitoring();
    } catch (err) {
      console.error('Extraction error:', err);
      alert('Error launching extraction: ' + err.message);
    }
  });

  function startLiveStreamMonitoring() {
    state.isProcessing = true;
    livePlaceholderBox.style.display = 'none';
    liveFrameImg.style.display = 'block';
    liveFrameImg.src = `/api/video/stream?t=${Date.now()}`;
    liveStreamBadge.textContent = 'Streaming Live';
    liveStreamBadge.className = 'badge success';
    telemetryStatusPill.textContent = 'Processing';
    telemetryStatusPill.className = 'badge success';

    if (state.telemetryInterval) clearInterval(state.telemetryInterval);
    state.telemetryInterval = setInterval(fetchTelemetry, 450);
  }

  async function fetchTelemetry() {
    try {
      const res = await fetch('/api/video/telemetry');
      const data = await res.json();

      statWindshieldsFound.textContent = data.windshields_count;
      statFramesDone.textContent = data.processed_count;
      statFramesTotal.textContent = `of ${data.total_frames_target} frames`;
      statFps.textContent = `${data.fps.toFixed(1)} fps`;
      statEta.textContent = `${data.eta_seconds}s`;

      badgeWindshieldsCount.textContent = `${data.windshields_count} Windshields Extracted`;

      if (data.is_processing) {
        setActivity('Extracting Windshields', data.status_message, data.is_paused ? 'PAUSED' : 'PROCESSING', data.progress);
      } else {
        // Extraction completed or stopped
        clearInterval(state.telemetryInterval);
        state.telemetryInterval = null;
        state.isProcessing = false;

        liveStreamBadge.textContent = 'Complete';
        liveStreamBadge.className = 'badge';
        telemetryStatusPill.textContent = 'Finished';
        telemetryStatusPill.className = 'badge';

        setActivity('Extraction Complete', data.status_message, 'DONE', 100);
        logConsole(`[Complete] Extraction finished. Found ${data.windshields_count} qualifying windshields.`);

        // Auto load results into inspector
        loadInspectorResults();
      }
    } catch (err) {
      console.error('Telemetry error:', err);
    }
  }

  btnPauseExtract.addEventListener('click', async () => {
    await fetch('/api/video/pause', { method: 'POST' });
    logConsole('[Control] Extraction paused.');
  });

  btnResumeExtract.addEventListener('click', async () => {
    await fetch('/api/video/resume', { method: 'POST' });
    logConsole('[Control] Extraction resumed.');
  });

  btnStopExtract.addEventListener('click', async () => {
    await fetch('/api/video/stop', { method: 'POST' });
    logConsole('[Control] Extraction stopped.');
  });

  btnGotoInspector.addEventListener('click', () => {
    switchTab('tab-inspector');
  });

  function logConsole(msg) {
    const p = document.createElement('div');
    p.textContent = msg;
    liveConsoleLog.appendChild(p);
    liveConsoleLog.scrollTop = liveConsoleLog.scrollHeight;
  }

  function setActivity(title, detail, stateText, pct) {
    activityTitle.textContent = title;
    activityDetail.textContent = detail;
    activityState.textContent = stateText;
    activityProgressBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    activityCard.setAttribute('aria-busy', stateText === 'PROCESSING' ? 'true' : 'false');
  }

  // --- TAB 4: COORDINATES INSPECTOR ---
  async function loadInspectorResults() {
    try {
      const res = await fetch('/api/video/results');
      const data = await res.json();

      state.windshieldRecords = data.windshield_records || [];
      state.recordsByFrame = data.records_by_frame || {};
      state.uniqueFramesWithDetections = Object.keys(state.recordsByFrame).map(Number).sort((a, b) => a - b);

      inspectorTotalCount.textContent = state.windshieldRecords.length;
      badgeWindshieldsCount.textContent = `${state.windshieldRecords.length} Windshields Extracted`;

      renderWindshieldsList();

      if (state.uniqueFramesWithDetections.length > 0) {
        if (!state.uniqueFramesWithDetections.includes(state.inspectorCurrentFrame)) {
          state.inspectorCurrentFrame = state.uniqueFramesWithDetections[0];
        }
        loadInspectorFrame(state.inspectorCurrentFrame);
      } else {
        inspectorFrameBadge.textContent = 'Frame: None';
      }
    } catch (err) {
      console.error('Inspector load error:', err);
    }
  }

  btnRefreshWindshields.addEventListener('click', loadInspectorResults);

  btnClearWindshields.addEventListener('click', async () => {
    if (!confirm('Clear all extracted windshield detections?')) return;
    await fetch('/api/clear-detections', { method: 'POST' });
    loadInspectorResults();
  });

  function renderWindshieldsList() {
    const filterText = filterWindshieldText.value.toLowerCase().trim();
    windshieldsContainer.innerHTML = '';

    const filtered = state.windshieldRecords.filter(item => {
      if (!filterText) return true;
      const v = (item.vehicle_type || '').toLowerCase();
      const f = `frame ${item.frame_idx}`;
      return v.includes(filterText) || f.includes(filterText);
    });

    if (filtered.length === 0) {
      windshieldsContainer.innerHTML = `
        <div style="padding: 2rem; text-align: center; color: var(--text-muted);">
          No windshields match your filter or none extracted yet.
        </div>`;
      return;
    }

    filtered.forEach(item => {
      const card = document.createElement('div');
      card.className = `plate-item ${state.selectedWindshieldId === item.id ? 'selected' : ''}`;
      card.id = `ws-card-${item.id}`;

      const wb = item.windshield_bbox || [0, 0, 0, 0];
      const yb = item.yolo_bbox || {};
      const confPct = Math.round(item.confidence * 100);

      // Format 4 corners
      const c = item.corners_px || [];
      const tl = c.find(p => p.corner === 'top_left') || { x: wb[0], y: wb[1] };
      const tr = c.find(p => p.corner === 'top_right') || { x: wb[2], y: wb[1] };
      const br = c.find(p => p.corner === 'bottom_right') || { x: wb[2], y: wb[3] };
      const bl = c.find(p => p.corner === 'bottom_left') || { x: wb[0], y: wb[3] };

      card.innerHTML = `
        <div class="plate-item-left">
          <img src="${item.crop_url}" class="windshield-crop-thumb" alt="Windshield" onerror="this.src='/static/placeholder.jpg'">
          <div>
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
              <span class="plate-text-display">${item.vehicle_type} Glass</span>
              <span class="badge ${confPct >= 30 ? 'success' : 'warning'}">${confPct}% Score</span>
            </div>
            <div class="plate-meta-info">
              Frame ${item.frame_idx} (${item.timestamp_sec}s) · Box: [${wb.join(', ')}] (${item.windshield_w}x${item.windshield_h} px)
            </div>
            <div style="margin-top: 0.35rem; display: flex; align-items: center; gap: 0.45rem;">
              <span class="plate-yolo-tag">${item.yolo_line}</span>
            </div>
            <div style="font-size: 0.70rem; color: var(--text-muted); margin-top: 0.2rem;">
              Corners: TL(${tl.x}, ${tl.y}) · TR(${tr.x}, ${tr.y}) · BR(${br.x}, ${br.y}) · BL(${bl.x}, ${bl.y})
            </div>
          </div>
        </div>
        <div style="display: flex; flex-direction: column; gap: 0.4rem; align-items: flex-end;">
          <button class="btn btn-secondary btn-sm btn-copy-yolo" data-yolo="${item.yolo_line}">
            Copy YOLO
          </button>
          <button class="btn btn-danger btn-sm btn-del-det" data-id="${item.id}" style="padding: 0.2rem 0.5rem; font-size: 0.72rem;">
            Delete
          </button>
        </div>
      `;

      card.addEventListener('click', e => {
        if (e.target.closest('button')) return;
        selectWindshield(item);
      });

      windshieldsContainer.appendChild(card);
    });

    // Attach button listeners
    windshieldsContainer.querySelectorAll('.btn-copy-yolo').forEach(b => {
      b.addEventListener('click', e => {
        e.stopPropagation();
        const yolo = b.getAttribute('data-yolo');
        navigator.clipboard.writeText(yolo);
        const originalText = b.textContent;
        b.textContent = 'Copied!';
        b.classList.add('btn-primary');
        setTimeout(() => {
          b.textContent = originalText;
          b.classList.remove('btn-primary');
        }, 1200);
      });
    });

    windshieldsContainer.querySelectorAll('.btn-del-det').forEach(b => {
      b.addEventListener('click', async e => {
        e.stopPropagation();
        const id = b.getAttribute('data-id');
        await fetch(`/api/delete-detection/${id}`, { method: 'DELETE' });
        loadInspectorResults();
      });
    });
  }

  filterWindshieldText.addEventListener('input', renderWindshieldsList);

  function selectWindshield(item) {
    state.selectedWindshieldId = item.id;
    inspectorYoloPreview.textContent = `YOLO: ${item.yolo_line}`;

    // Highlight card
    document.querySelectorAll('.plate-item').forEach(el => el.classList.remove('selected'));
    const targetCard = document.getElementById(`ws-card-${item.id}`);
    if (targetCard) targetCard.classList.add('selected');

    if (item.frame_idx !== state.inspectorCurrentFrame) {
      state.inspectorCurrentFrame = item.frame_idx;
      loadInspectorFrame(item.frame_idx);
    } else {
      redrawInspectorCanvas();
    }
  }

  function loadInspectorFrame(frameIdx) {
    if (!state.currentVideoPath) return;
    inspectorFrameBadge.textContent = `Frame: ${frameIdx}`;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = `/api/video/frame?path=${encodeURIComponent(state.currentVideoPath)}&frame_idx=${frameIdx}&t=${Date.now()}`;
    img.onload = () => {
      state.inspectorImage = img;
      redrawInspectorCanvas();
    };
  }

  function redrawInspectorCanvas() {
    if (!state.inspectorImage) return;

    const container = inspectorCanvas.parentElement;
    const cw = container.clientWidth || 640;
    const aspect = state.inspectorImage.naturalHeight / state.inspectorImage.naturalWidth;
    inspectorCanvas.width = cw;
    inspectorCanvas.height = Math.round(cw * aspect);

    const w = inspectorCanvas.width;
    const h = inspectorCanvas.height;

    // Draw background frame
    inspectorCtx.drawImage(state.inspectorImage, 0, 0, w, h);

    // Draw detections on this frame
    const dets = state.recordsByFrame[state.inspectorCurrentFrame] || [];

    dets.forEach(d => {
      const isSelected = d.id === state.selectedWindshieldId;
      const yb = d.yolo_bbox || {};

      // Calculate pixel box using normalized YOLO coordinates on the canvas
      let x1, y1, bw, bh;
      if (yb && typeof yb.x_center === 'number') {
        x1 = (yb.x_center - yb.width / 2.0) * w;
        y1 = (yb.y_center - yb.height / 2.0) * h;
        bw = yb.width * w;
        bh = yb.height * h;
      } else {
        const wb = d.windshield_bbox;
        const origW = state.videoMeta?.width || state.inspectorImage.naturalWidth;
        const origH = state.videoMeta?.height || state.inspectorImage.naturalHeight;
        x1 = (wb[0] / origW) * w;
        y1 = (wb[1] / origH) * h;
        bw = ((wb[2] - wb[0]) / origW) * w;
        bh = ((wb[3] - wb[1]) / origH) * h;
      }

      // Draw bounding box
      inspectorCtx.strokeStyle = isSelected ? '#f59e0b' : '#00e5ff';
      inspectorCtx.lineWidth = isSelected ? 3.5 : 2.0;
      inspectorCtx.strokeRect(x1, y1, bw, bh);

      if (isSelected) {
        inspectorCtx.fillStyle = 'rgba(245, 158, 11, 0.25)';
        inspectorCtx.fillRect(x1, y1, bw, bh);
      }

      // Draw Corner points using normalized coordinates
      const cornersNorm = d.corners_norm || [];
      if (cornersNorm.length > 0) {
        cornersNorm.forEach(cp => {
          const cx = cp.x * w;
          const cy = cp.y * h;
          inspectorCtx.beginPath();
          inspectorCtx.arc(cx, cy, 4, 0, Math.PI * 2);
          inspectorCtx.fillStyle = isSelected ? '#f59e0b' : '#10b981';
          inspectorCtx.fill();
        });
      }

      // Label
      const confPct = Math.round(d.confidence * 100);
      const text = `${d.vehicle_type} Glass ${confPct}%`;
      inspectorCtx.font = 'bold 11px sans-serif';
      const tw = inspectorCtx.measureText(text).width;

      inspectorCtx.fillStyle = isSelected ? '#f59e0b' : '#00e5ff';
      inspectorCtx.fillRect(x1, Math.max(0, y1 - 18), tw + 8, 18);

      inspectorCtx.fillStyle = '#0f172a';
      inspectorCtx.fillText(text, x1 + 4, Math.max(0, y1 - 5));
    });
  }

  // Click on inspector canvas to pick detection
  inspectorCanvas.addEventListener('click', e => {
    if (!state.inspectorImage) return;
    const rect = inspectorCanvas.getBoundingClientRect();
    const clickXNorm = (e.clientX - rect.left) / rect.width;
    const clickYNorm = (e.clientY - rect.top) / rect.height;

    const dets = state.recordsByFrame[state.inspectorCurrentFrame] || [];
    for (const d of dets) {
      const yb = d.yolo_bbox || {};
      let x1Norm, y1Norm, x2Norm, y2Norm;
      if (yb && typeof yb.x_center === 'number') {
        x1Norm = yb.x_center - yb.width / 2.0;
        y1Norm = yb.y_center - yb.height / 2.0;
        x2Norm = yb.x_center + yb.width / 2.0;
        y2Norm = yb.y_center + yb.height / 2.0;
      } else {
        const wb = d.windshield_bbox;
        const origW = state.videoMeta?.width || 1920;
        const origH = state.videoMeta?.height || 1080;
        x1Norm = wb[0] / origW;
        y1Norm = wb[1] / origH;
        x2Norm = wb[2] / origW;
        y2Norm = wb[3] / origH;
      }

      if (clickXNorm >= x1Norm && clickXNorm <= x2Norm && clickYNorm >= y1Norm && clickYNorm <= y2Norm) {
        selectWindshield(d);
        const card = document.getElementById(`ws-card-${d.id}`);
        if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        break;
      }
    }
  });

  // Prev / Next Frame with windshields
  btnPrevWindshieldFrame.addEventListener('click', () => {
    const list = state.uniqueFramesWithDetections;
    if (list.length === 0) return;
    const idx = list.indexOf(state.inspectorCurrentFrame);
    if (idx > 0) {
      state.inspectorCurrentFrame = list[idx - 1];
      loadInspectorFrame(state.inspectorCurrentFrame);
    } else {
      state.inspectorCurrentFrame = list[list.length - 1];
      loadInspectorFrame(state.inspectorCurrentFrame);
    }
  });

  btnNextWindshieldFrame.addEventListener('click', () => {
    const list = state.uniqueFramesWithDetections;
    if (list.length === 0) return;
    const idx = list.indexOf(state.inspectorCurrentFrame);
    if (idx >= 0 && idx < list.length - 1) {
      state.inspectorCurrentFrame = list[idx + 1];
      loadInspectorFrame(state.inspectorCurrentFrame);
    } else {
      state.inspectorCurrentFrame = list[0];
      loadInspectorFrame(state.inspectorCurrentFrame);
    }
  });

  // --- TAB 5: EXPORT CENTER ---
  exportSplitSlider.addEventListener('input', () => {
    const val = parseFloat(exportSplitSlider.value);
    const trainPct = Math.round(val * 100);
    const valPct = 100 - trainPct;
    exportSplitLabel.textContent = `${trainPct}% Train / ${valPct}% Val`;
    btnDownloadYoloZip.href = `/api/export/yolo?split=${val.toFixed(2)}`;
  });

  // Initialize
  loadVideoList();
});
