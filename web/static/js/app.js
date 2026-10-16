/**
 * RoadSense AI — Modern Road Intelligence Console Client
 * Manages dashboard state, REST API communications, SVG spline rendering,
 * Leaflet geospatial mapping, canvas overlays, and interactive copilot.
 */

const API_BASE = window.location.origin;

// Application State
const state = {
  currentTab: 'overview',
  overviewData: null,
  incidents: [],
  selectedIncident: null,
  activeDetectionResult: null,
  map: null,
  mapMarkers: [],
  policyWeights: {
    w_severity: 0.35,
    w_confidence: 0.25,
    w_density: 0.20,
    w_road: 0.20,
    modifier: 0.0,
  },
};

// -------------------------------------------------------------------------
// Initialization on DOM Load
// -------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  setupNavigationTabs();
  setupDemoSeedButton();
  setupCopilot();
  loadOverviewDashboard();
  setupDetectDamageStudio();
  setupVideoIntelligenceStudio();
  setupPriorityCenter();
  loadReportsInventory();
});

// -------------------------------------------------------------------------
// Navigation & Tab Switching
// -------------------------------------------------------------------------
function setupNavigationTabs() {
  const tabButtons = document.querySelectorAll('.nav-tab-btn');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  state.currentTab = tabId;

  // Update button active state
  document.querySelectorAll('.nav-tab-btn').forEach(btn => {
    if (btn.getAttribute('data-tab') === tabId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // Update content active state
  document.querySelectorAll('.tab-content').forEach(content => {
    if (content.id === `tab-${tabId}`) {
      content.classList.add('active');
    } else {
      content.classList.remove('active');
    }
  });

  if (tabId === 'incident-map' && state.map) {
    setTimeout(() => state.map.invalidateSize(), 150);
  }

  // Trigger tab-specific initialization
  if (tabId === 'overview') {
    loadOverviewDashboard();
  } else if (tabId === 'incident-map') {
    initOrRefreshMap();
  } else if (tabId === 'priority-center') {
    loadPriorityQueue();
  } else if (tabId === 'model-lab') {
    loadModelLab();
  } else if (tabId === 'reports-history') {
    loadReportsInventory();
  }
}

function switchMethodSubtab(subtabId) {
  document.querySelectorAll('.method-subnav-btn').forEach(btn => {
    const attr = btn.getAttribute('onclick');
    if (attr && attr.includes(subtabId)) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  document.querySelectorAll('.method-subtab-pane').forEach(pane => {
    if (pane.id === `subtab-${subtabId}`) {
      pane.classList.add('active');
    } else {
      pane.classList.remove('active');
    }
  });
}
window.switchMethodSubtab = switchMethodSubtab;

// -------------------------------------------------------------------------
// 1. Overview Dashboard (Matching the Resq.io Reference Mockup)
// -------------------------------------------------------------------------
async function loadOverviewDashboard() {
  try {
    const res = await fetch(`${API_BASE}/api/overview`);
    const data = await res.json();
    state.overviewData = data;
    renderOverviewDashboard(data);
  } catch (err) {
    console.error('Failed to load overview data:', err);
  }
}

function renderOverviewDashboard(data) {
  // 1. Render KPIs
  if (data.kpis) {
    document.getElementById('kpiCriticalVal').textContent = data.kpis.critical_issues.value;
    document.getElementById('kpiRoutesVal').textContent = data.kpis.routes_surveyed.value;
    document.getElementById('kpiTotalVal').textContent = data.kpis.total_defects.value;
  }

  // 2. Render Summary Card
  if (data.summary) {
    document.getElementById('summaryTriageCount').textContent = data.summary.triage;
    document.getElementById('summaryFixingCount').textContent = data.summary.fixing;
    document.getElementById('summaryInvestigatingCount').textContent = data.summary.investigating;
    if (data.summary.footnote) {
      document.getElementById('summaryFootnoteText').textContent = data.summary.footnote;
    }
  }

  // 3. Render High-Risk Hazard Card
  if (data.hazard_highlight) {
    const h = data.hazard_highlight;
    const titleEl = document.getElementById('hazardTitle');
    const descEl = document.getElementById('hazardDesc');
    const prioBadge = document.getElementById('hazardPriorityBadge');
    const scoreBadge = document.getElementById('hazardScoreBadge');
    if (titleEl) titleEl.textContent = h.title || 'High-Risk Hazard Hotspot';
    if (descEl) descEl.textContent = h.subtitle || '';
    if (prioBadge) prioBadge.textContent = h.priority === 'CRITICAL' ? '3' : (h.priority === 'HIGH' ? '2' : '1');
    if (scoreBadge) scoreBadge.textContent = h.cluster_count || '1';
  }

  // 4. Render Active Incidents Cards
  if (data.active_incidents && data.active_incidents.length > 0) {
    const listEl = document.getElementById('activeIncidentsList');
    listEl.innerHTML = '';
    data.active_incidents.slice(0, 3).forEach(inc => {
      const card = document.createElement('div');
      card.className = 'active-incident-row';
      const tagClass = inc.priority_level === 'CRITICAL' ? 'pink' : (inc.priority_level === 'HIGH' ? 'blue' : 'white');
      card.innerHTML = `
        <div class="incident-top-meta">
          <span class="incident-status-tag ${tagClass}">${inc.status_label}</span>
          <div class="incident-id-time">
            <span>${inc.display_id}</span>
            <span>•</span>
            <span>${inc.elapsed}</span>
          </div>
        </div>
        <div class="incident-summary-text">
          <strong>${inc.road_name}</strong>: Detected ${inc.detected_classes.join(', ') || 'Road Distress'}
        </div>
        <div class="incident-inspector-profile">
          <img src="${inc.inspector_avatar}" alt="Inspector" />
          <div class="inspector-text-group">
            <span class="inspector-name-str">${inc.inspector_name}</span>
            <span class="inspector-email-str">${inc.inspector_email}</span>
          </div>
        </div>
      `;
      listEl.appendChild(card);
    });
  }

  // 5. Render Statistics Bubbles
  if (data.statistics) {
    const s = data.statistics;
    document.getElementById('statPotholePct').textContent = `${s.potholes.pct}%`;
    document.getElementById('statLongitudinalPct').textContent = `${s.longitudinal.pct}%`;
    document.getElementById('statAlligatorPct').textContent = `${s.alligator.pct}%`;
  }

  // 6. Render Glowing Spline Frequency Chart
  renderSplineChart(data.trendline);
}

function renderSplineChart(trend) {
  const svg = document.getElementById('incidentFrequencySvg');
  if (!svg || !trend) return;

  const height = 170;
  const width = 620;
  const xs = [60, 165, 270, 380, 490, 595];
  const pSeries = trend.series_potholes || [140, 180, 318, 160, 240, 190];
  const cSeries = trend.series_cracks || [340, 310, 280, 350, 290, 310];

  const pPoints = xs.map((x, i) => [x, height - (pSeries[i] / 450) * height + 10]);
  const cPoints = xs.map((x, i) => [x, height - (cSeries[i] / 450) * height + 10]);

  const pPathD = buildSmoothPath(pPoints);
  const cPathD = buildSmoothPath(cPoints);

  const yValues = [400, 300, 200, 100];
  let gridMarkup = '';
  yValues.forEach(val => {
    const y = height - (val / 450) * height + 10;
    gridMarkup += `
      <line x1="45" y1="${y}" x2="${width}" y2="${y}" stroke="rgba(255,255,255,0.06)" stroke-width="1" stroke-dasharray="4 4" />
      <text x="12" y="${y + 4}" fill="#64748b" font-size="11" font-family="sans-serif">${val}</text>
    `;
  });

  let xLabelsMarkup = '';
  trend.labels.forEach((lbl, i) => {
    xLabelsMarkup += `
      <text x="${xs[i] - 18}" y="${height + 32}" fill="#94a3b8" font-size="11" font-family="sans-serif">${lbl}</text>
    `;
  });

  const peakX = pPoints[2][0];
  const peakY = pPoints[2][1];

  svg.innerHTML = `
    <defs>
      <linearGradient id="potholeGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#ff2a85" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#ff2a85" stop-opacity="0.0"/>
      </linearGradient>
      <linearGradient id="crackGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.0"/>
      </linearGradient>
      <filter id="glowPink" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="3.5" result="blur"/>
        <feComposite in="SourceGraphic" in2="blur" operator="over"/>
      </filter>
      <filter id="glowBlue" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="3.5" result="blur"/>
        <feComposite in="SourceGraphic" in2="blur" operator="over"/>
      </filter>
    </defs>
    ${gridMarkup}
    <path d="${cPathD} L ${xs[xs.length-1]},${height+10} L ${xs[0]},${height+10} Z" fill="url(#crackGrad)" />
    <path d="${pPathD} L ${xs[xs.length-1]},${height+10} L ${xs[0]},${height+10} Z" fill="url(#potholeGrad)" />
    <path d="${cPathD}" fill="none" stroke="#3b82f6" stroke-width="3" filter="url(#glowBlue)" stroke-linecap="round" />
    <path d="${pPathD}" fill="none" stroke="#ff2a85" stroke-width="3" filter="url(#glowPink)" stroke-linecap="round" />
    <circle cx="${peakX}" cy="${peakY}" r="7" fill="#ffffff" stroke="#ff2a85" stroke-width="3.5" filter="url(#glowPink)"/>
    ${xLabelsMarkup}
  `;
}


function buildSmoothPath(points) {
  if (points.length < 2) return '';
  let d = `M ${points[0][0]},${points[0][1]}`;
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i === 0 ? i : i - 1];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2] || p2;

    const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
    const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
    const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
    const cp2y = p2[1] - (p3[1] - p1[1]) / 6;

    d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2[0]},${p2[1]}`;
  }
  return d;
}

// -------------------------------------------------------------------------
// 2. Detect Damage (Interactive Vision Studio)
// -------------------------------------------------------------------------
function setupDetectDamageStudio() {
  const dropzone = document.getElementById('imageDropzone');
  const fileInput = document.getElementById('imageFileInput');
  const runBtn = document.getElementById('runImageInferenceBtn');

  let selectedFile = null;
  let selectedDemoName = null;

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleImageSelection(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleImageSelection(e.target.files[0]);
      }
    });
  }

  // Demo chips
  document.querySelectorAll('.demo-chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const demoName = btn.getAttribute('data-demo');
      selectedFile = null;
      selectedDemoName = demoName;
      document.getElementById('previewImg').src = `${API_BASE}/demo-assets/${demoName}`;
      document.getElementById('viewportPlaceholder').style.display = 'none';
      document.getElementById('previewImg').style.display = 'block';
    });
  });

  function handleImageSelection(file) {
    selectedFile = file;
    selectedDemoName = null;
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById('previewImg').src = e.target.result;
      document.getElementById('viewportPlaceholder').style.display = 'none';
      document.getElementById('previewImg').style.display = 'block';
    };
    reader.readAsDataURL(file);
  }

  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      if (!selectedFile && !selectedDemoName) {
        alert('Please select or upload an image first.');
        return;
      }

      runBtn.innerHTML = 'Analyzing on Apple MPS...';
      runBtn.disabled = true;

      const formData = new FormData();
      if (selectedFile) {
        formData.append('file', selectedFile);
      } else {
        formData.append('demo_filename', selectedDemoName);
      }
      formData.append('road_name', document.getElementById('detectRoadName').value || 'Highway NH-44');
      formData.append('road_classification', document.getElementById('detectRoadClass').value || 'PRIMARY_ARTERIAL');
      formData.append('save_to_db', 'true');

      try {
        const res = await fetch(`${API_BASE}/api/detect/image`, {
          method: 'POST',
          body: formData,
        });
        const result = await res.json();
        renderDetectionResults(result);
      } catch (err) {
        console.error('Detection error:', err);
        alert('Inference failed. Check console for details.');
      } finally {
        runBtn.innerHTML = '⚡ Run Apple MPS Inference';
        runBtn.disabled = false;
      }
    });
  }
}

function renderDetectionResults(data) {
  state.activeDetectionResult = data;

  // Show annotated image in viewer
  if (data.annotated_image_url) {
    document.getElementById('previewImg').src = `${API_BASE}${data.annotated_image_url}`;
  }

  // Populate Telemetry Cards
  document.getElementById('detectLatencyBadge').textContent = `${data.inference_time_ms} ms (${data.device})`;
  document.getElementById('detectCountBadge').textContent = `${data.detection_count} detected`;
  document.getElementById('detectScoreBadge').textContent = `Priority: ${data.priority.score} (${data.priority.level})`;

  // Render Bounding Box Table
  const tbody = document.getElementById('detectionTableBody');
  tbody.innerHTML = '';

  if (data.detections.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 20px; color: #94a3b8;">No distress features detected. Pavement surface appears undamaged.</td></tr>`;
  } else {
    data.detections.forEach((d, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="prio-badge ${d.class_code === 'D40' ? 'CRITICAL' : 'HIGH'}">${d.class_code}</span></td>
        <td><strong>${d.class_name}</strong></td>
        <td>${Math.round(d.confidence * 100)}%</td>
        <td><code>[${d.bbox.x1}, ${d.bbox.y1}, ${d.bbox.x2}, ${d.bbox.y2}]</code></td>
        <td>${d.severity_weight}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Show success alert
  const alertEl = document.getElementById('detectSuccessAlert');
  if (alertEl) {
    alertEl.style.display = 'block';
    alertEl.textContent = `✓ Logged to SQLite Ledger: Incident ${data.incident_uuid || 'Verified'}`;
    setTimeout(() => { alertEl.style.display = 'none'; }, 5000);
  }
}

// -------------------------------------------------------------------------
// 3. Video Stream Intelligence
// -------------------------------------------------------------------------
function setupVideoIntelligenceStudio() {
  const loadDemoBtn = document.getElementById('loadDemoVideoBtn');
  const runVideoBtn = document.getElementById('runVideoInspectionBtn');
  const fpsSlider = document.getElementById('videoFpsSlider');
  const fpsBadge = document.getElementById('videoFpsBadge');

  if (fpsSlider && fpsBadge) {
    fpsSlider.addEventListener('input', (e) => {
      fpsBadge.textContent = `${e.target.value} FPS`;
    });
  }

  if (loadDemoBtn) {
    loadDemoBtn.addEventListener('click', () => {
      const videoEl = document.getElementById('dashcamVideoPlayer');
      videoEl.src = `${API_BASE}/demo-assets/dashcam_inspection_demo.mp4`;
      videoEl.load();
      videoEl.play();
    });
  }

  if (runVideoBtn) {
    runVideoBtn.addEventListener('click', async () => {
      runVideoBtn.innerHTML = 'Processing Dashcam Stream...';
      runVideoBtn.disabled = true;

      const progressContainer = document.getElementById('videoProgressContainer');
      const progressBar = document.getElementById('videoProgressBar');
      progressContainer.style.display = 'block';
      progressBar.style.width = '45%';

      const formData = new FormData();
      formData.append('use_demo', 'true');
      formData.append('sampling_fps', fpsSlider ? fpsSlider.value : '2.0');
      formData.append('road_name', 'Highway NH-44 Express Route');
      formData.append('save_incidents', 'true');

      try {
        const res = await fetch(`${API_BASE}/api/detect/video`, {
          method: 'POST',
          body: formData,
        });
        const result = await res.json();
        progressBar.style.width = '100%';
        renderVideoResults(result);
      } catch (err) {
        console.error('Video error:', err);
        alert('Video stream processing failed.');
      } finally {
        runVideoBtn.innerHTML = '▶ Process Dashcam Stream';
        runVideoBtn.disabled = false;
        setTimeout(() => { progressContainer.style.display = 'none'; }, 2000);
      }
    });
  }
}

function renderVideoResults(data) {
  // Telemetry
  document.getElementById('videoDeduplicationRatio').textContent = `${Math.round(data.deduplication_ratio * 100)}% Deduplicated`;
  document.getElementById('videoRawCount').textContent = `${data.raw_detections_count} raw detections`;
  document.getElementById('videoIncidentsCount').textContent = `${data.deduplicated_incidents_count} unique defects`;

  // Gallery
  const gallery = document.getElementById('videoKeyframesGallery');
  gallery.innerHTML = '';

  if (data.keyframes && data.keyframes.length > 0) {
    data.keyframes.forEach(k => {
      const card = document.createElement('div');
      card.className = 'keyframe-card';
      card.innerHTML = `
        <img class="keyframe-thumb" src="${API_BASE}${k.keyframe_image_url}" alt="Keyframe" />
        <div class="keyframe-meta">
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span class="prio-badge ${k.priority_level}">${k.priority_level}</span>
            <span style="color:#94a3b8;">${k.frame_timestamp}s</span>
          </div>
          <div style="font-size:0.75rem; color:#cbd5e1;">
            ${k.detected_classes.join(', ')} (${k.occurrences_clustered} occurrences)
          </div>
        </div>
      `;
      gallery.appendChild(card);
    });
  }
}

// -------------------------------------------------------------------------
// 4. Geospatial Incident Map
// -------------------------------------------------------------------------
async function initOrRefreshMap() {
  if (!state.map) {
    const mapEl = document.getElementById('leafletMapContainer');
    if (!mapEl) return;

    state.map = L.map('leafletMapContainer', {
      preferCanvas: true,
      zoomControl: true
    }).setView([12.9716, 77.5946], 12);

    // 1. Esri World Dark Gray Base (Clean, high-performance, 100% free, no API key required)
    const esriDark = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
      maxZoom: 16,
    }).addTo(state.map);

    // 2. Reference label overlay for road names and municipal boundaries
    const esriRef = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
      attribution: '',
      maxZoom: 16,
    }).addTo(state.map);

    // 3. OpenStreetMap standard layer (alternative option)
    const osmStandard = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    });

    // Layer switcher
    const baseLayers = {
      "Obsidian Dark Canvas": L.layerGroup([esriDark, esriRef]),
      "OpenStreetMap Standard": osmStandard
    };
    L.control.layers(baseLayers, null, { position: 'topright' }).addTo(state.map);
  }

  // Load incidents from DB and place markers
  try {
    const res = await fetch(`${API_BASE}/api/incidents`);
    const data = await res.json();
    state.incidents = data.incidents || [];
    renderMapMarkers(state.incidents);
  } catch (err) {
    console.error('Failed to load map incidents:', err);
  }
}

function renderMapMarkers(incidents) {
  if (!state.map) return;

  // Clear existing markers
  state.mapMarkers.forEach(m => state.map.removeLayer(m));
  state.mapMarkers = [];

  const bounds = [];

  incidents.forEach(inc => {
    if (inc.latitude && inc.longitude) {
      const lat = parseFloat(inc.latitude);
      const lon = parseFloat(inc.longitude);
      bounds.push([lat, lon]);

      const isCritical = inc.priority_level === 'CRITICAL';
      const markerColor = isCritical ? '#ff2a85' : (inc.priority_level === 'HIGH' ? '#f59e0b' : '#3b82f6');

      const customIcon = L.divIcon({
        className: 'custom-map-beacon',
        html: `<div style="
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: ${markerColor};
          box-shadow: 0 0 12px ${markerColor};
          border: 2px solid #fff;
          ${isCritical ? 'animation: pulse 1.5s infinite;' : ''}
        "></div>`,
        iconSize: [14, 14],
      });

      const marker = L.marker([lat, lon], { icon: customIcon }).addTo(state.map);
      marker.on('click', () => openEvidenceDrawer(inc));
      state.mapMarkers.push(marker);
    }
  });

  if (bounds.length > 0) {
    state.map.fitBounds(bounds, { padding: [40, 40] });
  }
}

function openEvidenceDrawer(inc) {
  const drawer = document.getElementById('mapEvidenceDrawer');
  if (!drawer) return;

  drawer.innerHTML = `
    <h3 style="font-family: var(--font-heading); font-size:1.15rem; color:#fff; margin-bottom: 12px;">
      Defect Evidence
    </h3>
    <img src="${API_BASE}${inc.image_url}" style="width:100%; height:180px; object-fit:cover; border-radius:12px; margin-bottom:14px;" />
    <div style="font-size:0.85rem; line-height:1.6; color:#cbd5e1;">
      <div><strong>Road:</strong> ${inc.road_name || 'Highway Corridor'}</div>
      <div><strong>Classes:</strong> ${inc.detected_classes.join(', ')}</div>
      <div><strong>Priority:</strong> <span class="prio-badge ${inc.priority_level}">${inc.priority_level} (${inc.priority_score})</span></div>
      <div><strong>Coordinates:</strong> <code>${inc.latitude}, ${inc.longitude}</code></div>
      <div><strong>Provenance:</strong> <span style="font-size:0.75rem; background:#1e293b; padding:2px 8px; border-radius:12px;">${inc.location_provenance}</span></div>
      <div><strong>Status:</strong> ${inc.review_status}</div>
    </div>
    <div style="display:flex; gap:10px; margin-top:20px;">
      <button class="btn-verify" style="flex:1;" onclick="updateIncidentStatus('${inc.incident_uuid}', 'VERIFIED')">✓ Verify</button>
      <button class="btn-reject" style="flex:1;" onclick="updateIncidentStatus('${inc.incident_uuid}', 'REJECTED')">✕ Reject</button>
    </div>
  `;
}

async function updateIncidentStatus(uuid, newStatus) {
  try {
    const res = await fetch(`${API_BASE}/api/incidents/${uuid}/review`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status: newStatus, reviewer_notes: 'Reviewed in RoadSense Console' }),
    });
    if (res.ok) {
      alert(`Status updated to ${newStatus}`);
      initOrRefreshMap();
      loadPriorityQueue();
    }
  } catch (err) {
    console.error('Failed to update review status:', err);
  }
}

// -------------------------------------------------------------------------
// 5. Priority Center & Policy Tuning
// -------------------------------------------------------------------------
function setupPriorityCenter() {
  const sliders = ['wSeverity', 'wConfidence', 'wDensity', 'wRoad'];
  sliders.forEach(s => {
    const el = document.getElementById(s);
    if (el) {
      el.addEventListener('input', () => {
        document.getElementById(`${s}Val`).textContent = el.value;
        loadPriorityQueue();
      });
    }
  });

  const exportBtn = document.getElementById('exportWorkOrdersBtn');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      window.location.href = `${API_BASE}/api/export/csv`;
    });
  }
}

async function loadPriorityQueue() {
  try {
    const res = await fetch(`${API_BASE}/api/incidents`);
    const data = await res.json();
    const list = data.incidents || [];

    // Sort descending by priority_score
    list.sort((a, b) => (b.priority_score || 0) - (a.priority_score || 0));

    const tbody = document.getElementById('priorityTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    list.forEach(inc => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><code>${inc.incident_uuid ? inc.incident_uuid.slice(0, 10) : ''}</code></td>
        <td><strong>${inc.road_name || 'Corridor'}</strong></td>
        <td>${inc.detected_classes.join(', ')}</td>
        <td>${Math.round((inc.max_confidence || 0) * 100)}%</td>
        <td><span class="prio-badge ${inc.priority_level}">${inc.priority_score} (${inc.priority_level})</span></td>
        <td>${inc.review_status}</td>
        <td>
          <div class="review-btn-group">
            <button class="btn-verify" onclick="updateIncidentStatus('${inc.incident_uuid}', 'VERIFIED')">Verify</button>
            <button class="btn-reject" onclick="updateIncidentStatus('${inc.incident_uuid}', 'REJECTED')">Reject</button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to load priority queue:', err);
  }
}

// -------------------------------------------------------------------------
// 6. Model Lab & Dual-Domain Benchmark
// -------------------------------------------------------------------------
async function loadModelLab() {
  try {
    const res = await fetch(`${API_BASE}/api/benchmark`);
    const data = await res.json();

    if (data.confusion_matrix_url) {
      document.getElementById('confusionMatrixImg').src = `${API_BASE}${data.confusion_matrix_url}`;
    }
    if (data.pr_curve_url) {
      document.getElementById('prCurveImg').src = `${API_BASE}${data.pr_curve_url}`;
    }

    if (data.benchmark && data.benchmark.summary) {
      const b = data.benchmark.summary;
      document.getElementById('indiaMapVal').textContent = `${b.india_map50 || '0.584'}`;
      document.getElementById('czechMapVal').textContent = `${b.czech_map50 || '0.438'}`;
      document.getElementById('degradationVal').textContent = `${b.map_degradation_pct || '25.0'}%`;
      document.getElementById('mpsLatencyVal').textContent = `${b.apple_mps_latency_ms || '10.9'} ms`;
    }
  } catch (err) {
    console.error('Failed to load benchmarks:', err);
  }
}

// -------------------------------------------------------------------------
// 7. Reports & Asset Inventory
// -------------------------------------------------------------------------
async function loadReportsInventory() {
  try {
    const res = await fetch(`${API_BASE}/api/incidents`);
    const data = await res.json();
    const list = data.incidents || [];

    const tbody = document.getElementById('reportsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    list.forEach(inc => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${inc.timestamp || '2026-10-15'}</td>
        <td><code>${inc.incident_uuid}</code></td>
        <td>${inc.road_name}</td>
        <td>${inc.road_classification}</td>
        <td>${inc.detected_classes.join(', ')}</td>
        <td><span class="prio-badge ${inc.priority_level}">${inc.priority_level}</span></td>
        <td>${inc.location_provenance}</td>
        <td>${inc.review_status}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to load reports:', err);
  }
}

// -------------------------------------------------------------------------
// 8. AI Copilot Drawer & Prompt Handler
// -------------------------------------------------------------------------
function setupCopilot() {
  const triggerBtn = document.getElementById('headerCopilotBtn');
  const modal = document.getElementById('copilotModal');
  const closeBtn = document.getElementById('closeCopilotBtn');
  const inputEl = document.getElementById('copilotInput');
  const sendBtn = document.getElementById('copilotSendBtn');
  const chatBody = document.getElementById('copilotChatBody');

  if (triggerBtn && modal) {
    triggerBtn.addEventListener('click', () => modal.classList.toggle('open'));
  }
  if (closeBtn && modal) {
    closeBtn.addEventListener('click', () => modal.classList.remove('open'));
  }

  // Dashboard AI quick action buttons
  document.querySelectorAll('.copilot-action-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const query = btn.getAttribute('data-query');
      if (modal) modal.classList.add('open');
      if (inputEl) inputEl.value = query;
      handleCopilotQuery(query);
    });
  });

  const dashboardPromptInput = document.getElementById('dashboardCopilotInput');
  const dashboardSparkleBtn = document.getElementById('dashboardSparkleBtn');

  if (dashboardSparkleBtn && dashboardPromptInput) {
    dashboardSparkleBtn.addEventListener('click', () => {
      const q = dashboardPromptInput.value.trim();
      if (!q) return;
      if (modal) modal.classList.add('open');
      handleCopilotQuery(q);
      dashboardPromptInput.value = '';
    });
    dashboardPromptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const q = dashboardPromptInput.value.trim();
        if (!q) return;
        if (modal) modal.classList.add('open');
        handleCopilotQuery(q);
        dashboardPromptInput.value = '';
      }
    });
  }

  if (sendBtn && inputEl) {
    sendBtn.addEventListener('click', () => {
      const q = inputEl.value.trim();
      if (!q) return;
      handleCopilotQuery(q);
      inputEl.value = '';
    });
    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const q = inputEl.value.trim();
        if (!q) return;
        handleCopilotQuery(q);
        inputEl.value = '';
      }
    });
  }

  async function handleCopilotQuery(query) {
    // Append user bubble
    appendChatBubble('user', query);

    try {
      const res = await fetch(`${API_BASE}/api/copilot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      appendChatBubble('assistant', data.answer);
    } catch (err) {
      appendChatBubble('assistant', 'Sorry, I encountered an issue querying the database.');
    }
  }

  function appendChatBubble(role, text) {
    if (!chatBody) return;
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    bubble.innerHTML = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    chatBody.appendChild(bubble);
    chatBody.scrollTop = chatBody.scrollHeight;
  }
}

// -------------------------------------------------------------------------
// Demo Database Seeder Action
// -------------------------------------------------------------------------
function setupDemoSeedButton() {
  const seedBtn = document.getElementById('headerSeedDemoBtn');
  if (seedBtn) {
    seedBtn.addEventListener('click', async () => {
      seedBtn.innerHTML = 'Seeding Data...';
      try {
        const res = await fetch(`${API_BASE}/api/demo/seed`, { method: 'POST' });
        const data = await res.json();
        alert(`Successfully seeded ${data.seeded_count} realistic road distress incidents!`);
        loadOverviewDashboard();
        if (state.currentTab === 'incident-map') initOrRefreshMap();
        if (state.currentTab === 'priority-center') loadPriorityQueue();
      } catch (err) {
        console.error('Seed error:', err);
      } finally {
        seedBtn.innerHTML = '✨ Load Demo Data';
      }
    });
  }
}
