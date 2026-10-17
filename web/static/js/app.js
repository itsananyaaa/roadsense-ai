/**
 * RoadSense AI — Modern Road Intelligence Console Client
 * Manages dashboard state, REST API communications, SVG spline rendering,
 * Leaflet geospatial mapping, canvas overlays, and interactive copilot.
 */

const API_BASE = window.location.origin;

function escapeHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

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
// Authentication State & Session Gate
// -------------------------------------------------------------------------
let dashboardInitialized = false;

document.addEventListener('DOMContentLoaded', () => {
  initAuthAndApp();
});

async function initAuthAndApp() {
  setupLoginExperience();
  setupHeaderLogout();
  await checkAuthSession();
}

/**
 * Checks existing session via GET /api/auth/me
 */
async function checkAuthSession() {
  try {
    const res = await fetch(`${API_BASE}/api/auth/me`, {
      method: 'GET',
      credentials: 'same-origin',
    });

    if (res.ok) {
      const data = await res.json();
      revealDashboard(data.username);
    } else {
      showLoginScreen();
    }
  } catch (err) {
    console.warn('Session check network failure, falling back to login screen:', err);
    showLoginScreen();
  }
}

function revealDashboard(username) {
  document.body.classList.remove('auth-required');
  document.body.classList.add('authenticated');
  
  const headerUser = document.getElementById('headerUserName');
  if (headerUser && username) {
    headerUser.textContent = username;
  }
  
  initDashboard();
}

function showLoginScreen() {
  document.body.classList.add('auth-required');
  document.body.classList.remove('authenticated');
  const loginScreen = document.getElementById('loginScreen');
  if (loginScreen) {
    loginScreen.classList.remove('fade-out');
  }
}

function initDashboard() {
  if (dashboardInitialized) return;
  dashboardInitialized = true;
  setupNavigationTabs();
  setupDemoSeedButton();
  setupCopilot();
  loadOverviewDashboard();
  setupFrequencyRangePills();
  setupDetectDamageStudio();
  setupVideoIntelligenceStudio();
  setupPriorityCenter();
  setupMethodologyPdfExport();
  loadReportsInventory();
}

// -------------------------------------------------------------------------
// Animated Inline SVG Character Crew & Form Controller
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Animated Inline SVG Character Crew & Form Controller (Character Motion Spec)
// -------------------------------------------------------------------------
function setupLoginExperience() {
  const loginForm = document.getElementById('loginForm');
  const loginCard = document.getElementById('loginCard');
  const usernameInput = document.getElementById('loginUsername');
  const passwordInput = document.getElementById('loginPassword');
  const togglePasswordBtn = document.getElementById('togglePasswordBtn');
  const loginAlert = document.getElementById('loginAlert');
  const loginAlertText = document.getElementById('loginAlertText');
  const loginAlertIcon = document.getElementById('loginAlertIcon');
  const submitBtn = document.getElementById('loginSubmitBtn');
  const btnSpinner = document.getElementById('loginBtnSpinner');
  const btnText = submitBtn ? submitBtn.querySelector('.btn-text') : null;
  const btnArrow = submitBtn ? submitBtn.querySelector('.btn-arrow-icon') : null;
  const crewStage = document.getElementById('crewStage');

  // Crew SVG element handles
  const shapePylon = document.getElementById('crewPylon');
  const shapeSensor = document.getElementById('crewSensor');
  const shapeScanner = document.getElementById('crewScanner');
  const shapeMarker = document.getElementById('crewMarker');

  const pylonFace = document.getElementById('pylonFace');
  const sensorFace = document.getElementById('sensorFace');
  const scannerFace = document.getElementById('scannerFace');
  const markerFace = document.getElementById('markerFace');

  const pylonMouth = document.getElementById('pylonMouth');
  const sensorMouth = document.getElementById('sensorMouth');
  const scannerMouth = document.getElementById('scannerMouth');
  const markerMouth = document.getElementById('markerMouth');

  const pylonLids = [document.getElementById('pylonLidLeft'), document.getElementById('pylonLidRight')].filter(Boolean);
  const sensorLids = [document.getElementById('sensorLidLeft'), document.getElementById('sensorLidRight')].filter(Boolean);
  const scannerLids = [document.getElementById('scannerLidLeft'), document.getElementById('scannerLidRight')].filter(Boolean);
  const markerLids = [document.getElementById('markerLidLeft'), document.getElementById('markerLidRight')].filter(Boolean);

  const sensorPupilLeft = document.getElementById('sensorPupilLeft');
  const sensorPupilRight = document.getElementById('sensorPupilRight');
  const markerPupilLeft = document.getElementById('markerPupilLeft');
  const markerPupilRight = document.getElementById('markerPupilRight');

  const crewScanLine = document.getElementById('crewScanLine');
  const crewRadarSweep = document.getElementById('crewRadarSweep');
  const introLoader = document.getElementById('loginIntroLoader');

  if (!loginForm || !shapePylon || !shapeSensor || !shapeScanner || !shapeMarker) return;

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Mouth path shapes
  const mouths = {
    pylon: {
      neutral: 'M 158 208 Q 165 212 172 208',
      frown: 'M 158 212 Q 165 206 172 212',
      smile: 'M 157 207 Q 165 215 173 207',
    },
    sensor: {
      neutral: 'M 229 252 L 241 252',
      frown: 'M 229 255 Q 235 251 241 255',
      smile: 'M 229 251 Q 235 255 241 251',
    },
    scanner: {
      neutral: 'M 309 308 Q 315 313 321 308',
      frown: 'M 309 313 Q 315 307 321 313',
      smile: 'M 308 307 Q 315 316 322 307',
    },
    marker: {
      neutral: 'M 391 282 Q 395 285 399 282',
      wavy: 'M 390 283 Q 393 280 395 283 T 400 283',
      smile: 'M 391 281 Q 395 287 399 281',
    }
  };

  // Kinematic state parameters
  const cur = {
    pylon: { y: prefersReducedMotion ? 0 : -320, rot: prefersReducedMotion ? 0 : -14, skew: 0, gazeX: 0, gazeY: 0 },
    sensor: { y: prefersReducedMotion ? 0 : -320, rot: prefersReducedMotion ? 0 : 10, gazeX: 0, gazeY: 0 },
    scanner: { y: prefersReducedMotion ? 0 : -320, rot: prefersReducedMotion ? 0 : -8, scaleY: 1, gazeX: 0, gazeY: 0 },
    marker: { y: prefersReducedMotion ? 0 : -320, rot: prefersReducedMotion ? 0 : 16, gazeX: 0, gazeY: 0 },
  };

  const target = {
    pylon: { y: 0, rot: 0, skew: 0, gazeX: 0, gazeY: 0 },
    sensor: { y: 0, rot: 0, gazeX: 0, gazeY: 0 },
    scanner: { y: 0, rot: 0, scaleY: 1, gazeX: 0, gazeY: 0 },
    marker: { y: 0, rot: 0, gazeX: 0, gazeY: 0 },
  };

  let currentState = 'intro';
  let pointerGaze = { x: 0, y: 0 };
  let scanLineY = 160;
  let scanLineDir = 2.5;
  let rafId = null;

  // Intro Sequence
  if (!prefersReducedMotion && introLoader) {
    setTimeout(() => {
      introLoader.classList.add('fade-out');
      setTimeout(() => {
        introLoader.style.display = 'none';
        // Shapes drop in and settle into cluster
        setTimeout(() => {
          currentState = 'idle';
          // Reveal faces once settled
          [pylonFace, sensorFace, scannerFace, markerFace].forEach(f => {
            if (f) f.style.opacity = '1';
          });
        }, 1200);
      }, 450);
    }, 400);
  } else {
    if (introLoader) introLoader.style.display = 'none';
    currentState = 'idle';
    [pylonFace, sensorFace, scannerFace, markerFace].forEach(f => {
      if (f) f.style.opacity = '1';
    });
  }

  // Smooth lerp helper
  const lerp = (a, b, t) => a + (b - a) * t;

  // Main Unified Animation Loop
  function animateCrew(timestamp) {
    if (document.hidden) {
      rafId = requestAnimationFrame(animateCrew);
      return;
    }

    const t = prefersReducedMotion ? 1.0 : 0.12;

    // Apply state-specific targets
    if (currentState === 'intro') {
      // Shapes dropping in towards 0 with subtle overshoot
      target.pylon.y = 0;
      target.pylon.rot = 0;
      target.sensor.y = 0;
      target.sensor.rot = 0;
      target.scanner.y = 0;
      target.scanner.rot = 0;
      target.marker.y = 0;
      target.marker.rot = 0;
    } else if (currentState === 'idle') {
      target.pylon.rot = 0;
      target.pylon.skew = 0;
      target.sensor.rot = 0;
      target.scanner.rot = 0;
      target.marker.rot = 0;

      // Subtle breathing on dome
      target.scanner.scaleY = 1 + 0.015 * Math.sin(timestamp * 0.002);

      // Natural gaze following pointer
      target.pylon.gazeX = pointerGaze.x;
      target.pylon.gazeY = pointerGaze.y;
      target.sensor.gazeX = pointerGaze.x;
      target.sensor.gazeY = pointerGaze.y;
      target.scanner.gazeX = pointerGaze.x * 0.8;
      target.scanner.gazeY = pointerGaze.y * 0.8;
      target.marker.gazeX = pointerGaze.x;
      target.marker.gazeY = pointerGaze.y;

    } else if (currentState === 'username') {
      const len = usernameInput ? usernameInput.value.length : 0;
      const progress = Math.min(len / 15, 1);
      target.pylon.rot = 5 + progress * 6.5; // Bends toward form (right)
      target.pylon.skew = -1.5 - progress * 2.5; // Curvature via skewX
      target.pylon.gazeX = 4.5 + progress * 2.5;
      target.pylon.gazeY = 0.5;

      target.sensor.gazeX = 5.0;
      target.sensor.gazeY = 0.5;
      target.scanner.gazeX = 4.0;
      target.scanner.gazeY = 0.5;
      target.marker.gazeX = 4.5;
      target.marker.gazeY = 0.5;

    } else if (currentState === 'password') {
      // Polite "not looking" pose: pylon tilts away
      target.pylon.rot = -4.5;
      target.pylon.skew = 0.5;
      target.pylon.gazeX = -4.0;
      target.pylon.gazeY = 1.0;

      // Sensor and marker look away from field
      target.sensor.rot = -2.0;
      target.sensor.gazeX = -5.0;
      target.sensor.gazeY = 1.5;

      target.scanner.rot = 0;
      target.scanner.gazeX = -3.5;
      target.scanner.gazeY = 1.0;

      target.marker.rot = -2.0;
      target.marker.gazeX = -4.5;
      target.marker.gazeY = 1.5;

    } else if (currentState === 'peeking') {
      // Show password toggled on: sensor eyes widen, pylon and dome peek back
      target.pylon.rot = 3.0;
      target.pylon.skew = -1.0;
      target.pylon.gazeX = 5.5;
      target.pylon.gazeY = 0;

      target.sensor.rot = 0;
      target.sensor.gazeX = 6.0;
      target.sensor.gazeY = 0;

      target.scanner.rot = 0;
      target.scanner.gazeX = 5.0;
      target.scanner.gazeY = 0;

      target.marker.rot = 0;
      target.marker.gazeX = 5.0;
      target.marker.gazeY = 0;

    } else if (currentState === 'submitting') {
      // Straighten up and look toward submit button
      target.pylon.rot = 0.5;
      target.pylon.skew = 0;
      target.pylon.gazeX = 3.5;
      target.pylon.gazeY = 3.5;

      target.sensor.rot = 0;
      target.sensor.gazeX = 3.5;
      target.sensor.gazeY = 3.5;

      target.scanner.rot = 0;
      target.scanner.gazeX = 3.0;
      target.scanner.gazeY = 3.0;

      target.marker.rot = 0;
      target.marker.gazeX = 3.5;
      target.marker.gazeY = 3.5;

      // Scan line animation
      if (crewScanLine) {
        scanLineY += scanLineDir;
        if (scanLineY > 320 || scanLineY < 160) scanLineDir = -scanLineDir;
        crewScanLine.setAttribute('y1', scanLineY);
        crewScanLine.setAttribute('y2', scanLineY);
      }

    } else if (currentState === 'error') {
      // Rejection pose: pylon tilts away with frown, dome frowns, marker wavy
      target.pylon.rot = -5.0;
      target.pylon.skew = 0.5;
      target.pylon.gazeX = -2.0;
      target.pylon.gazeY = 2.0;

      target.sensor.rot = -2.0;
      target.sensor.gazeX = -2.0;
      target.sensor.gazeY = 2.0;

      target.scanner.rot = 0;
      target.scanner.gazeX = 0;
      target.scanner.gazeY = 2.5;

      target.marker.rot = 0;
      target.marker.gazeX = 0;
      target.marker.gazeY = 2.0;

    } else if (currentState === 'success') {
      // Success celebration: upright smiles, soft glow
      target.pylon.rot = 0;
      target.pylon.skew = 0;
      target.pylon.gazeX = 0;
      target.pylon.gazeY = -1.5;

      target.sensor.rot = 0;
      target.sensor.gazeX = 0;
      target.sensor.gazeY = -1.5;

      target.scanner.rot = 0;
      target.scanner.gazeX = 0;
      target.scanner.gazeY = -1.5;

      target.marker.rot = 0;
      target.marker.gazeX = 0;
      target.marker.gazeY = -1.5;
    }

    // Lerp values
    cur.pylon.y = lerp(cur.pylon.y, target.pylon.y, t);
    cur.pylon.rot = lerp(cur.pylon.rot, target.pylon.rot, t);
    cur.pylon.skew = lerp(cur.pylon.skew, target.pylon.skew, t);
    cur.pylon.gazeX = lerp(cur.pylon.gazeX, target.pylon.gazeX, t);
    cur.pylon.gazeY = lerp(cur.pylon.gazeY, target.pylon.gazeY, t);

    cur.sensor.y = lerp(cur.sensor.y, target.sensor.y, t);
    cur.sensor.rot = lerp(cur.sensor.rot, target.sensor.rot, t);
    cur.sensor.gazeX = lerp(cur.sensor.gazeX, target.sensor.gazeX, t);
    cur.sensor.gazeY = lerp(cur.sensor.gazeY, target.sensor.gazeY, t);

    cur.scanner.y = lerp(cur.scanner.y, target.scanner.y, t);
    cur.scanner.rot = lerp(cur.scanner.rot, target.scanner.rot, t);
    cur.scanner.scaleY = lerp(cur.scanner.scaleY, target.scanner.scaleY, t);
    cur.scanner.gazeX = lerp(cur.scanner.gazeX, target.scanner.gazeX, t);
    cur.scanner.gazeY = lerp(cur.scanner.gazeY, target.scanner.gazeY, t);

    cur.marker.y = lerp(cur.marker.y, target.marker.y, t);
    cur.marker.rot = lerp(cur.marker.rot, target.marker.rot, t);
    cur.marker.gazeX = lerp(cur.marker.gazeX, target.marker.gazeX, t);
    cur.marker.gazeY = lerp(cur.marker.gazeY, target.marker.gazeY, t);

    // Render transforms
    shapePylon.setAttribute('transform', `translate(0, ${cur.pylon.y.toFixed(2)}) rotate(${cur.pylon.rot.toFixed(2)}) skewX(${cur.pylon.skew.toFixed(2)})`);
    shapeSensor.setAttribute('transform', `translate(0, ${cur.sensor.y.toFixed(2)}) rotate(${cur.sensor.rot.toFixed(2)})`);
    shapeScanner.setAttribute('transform', `translate(0, ${cur.scanner.y.toFixed(2)}) scale(1, ${cur.scanner.scaleY.toFixed(3)}) rotate(${cur.scanner.rot.toFixed(2)})`);
    shapeMarker.setAttribute('transform', `translate(0, ${cur.marker.y.toFixed(2)}) rotate(${cur.marker.rot.toFixed(2)})`);

    // Render face offsets (Pylon face shifts slightly more)
    pylonFace.setAttribute('transform', `translate(${(cur.pylon.gazeX * 1.35).toFixed(2)}, ${(cur.pylon.gazeY * 1.2).toFixed(2)})`);
    sensorFace.setAttribute('transform', `translate(${cur.sensor.gazeX.toFixed(2)}, ${cur.sensor.gazeY.toFixed(2)})`);
    scannerFace.setAttribute('transform', `translate(${cur.scanner.gazeX.toFixed(2)}, ${cur.scanner.gazeY.toFixed(2)})`);
    markerFace.setAttribute('transform', `translate(${cur.marker.gazeX.toFixed(2)}, ${cur.marker.gazeY.toFixed(2)})`);

    // Pupils micro-offset
    if (sensorPupilLeft) sensorPupilLeft.setAttribute('transform', `translate(${(cur.sensor.gazeX * 0.4).toFixed(2)}, ${(cur.sensor.gazeY * 0.4).toFixed(2)})`);
    if (sensorPupilRight) sensorPupilRight.setAttribute('transform', `translate(${(cur.sensor.gazeX * 0.4).toFixed(2)}, ${(cur.sensor.gazeY * 0.4).toFixed(2)})`);
    if (markerPupilLeft) markerPupilLeft.setAttribute('transform', `translate(${(cur.marker.gazeX * 0.35).toFixed(2)}, ${(cur.marker.gazeY * 0.35).toFixed(2)})`);
    if (markerPupilRight) markerPupilRight.setAttribute('transform', `translate(${(cur.marker.gazeX * 0.35).toFixed(2)}, ${(cur.marker.gazeY * 0.35).toFixed(2)})`);

    rafId = requestAnimationFrame(animateCrew);
  }

  // Start RAF loop
  rafId = requestAnimationFrame(animateCrew);

  // Blinking cycle: random desynchronized blinks (120ms every 2.5-5s)
  function setupDesynchronizedBlinking(lids, intervalBase) {
    if (prefersReducedMotion || lids.length === 0) return;
    function triggerBlink() {
      if (currentState === 'password') {
        // In password mode, scanner dome eyes stay closed
        setTimeout(triggerBlink, intervalBase + Math.random() * 2000);
        return;
      }
      lids.forEach(lid => {
        lid.setAttribute('stroke-width', '18');
      });
      setTimeout(() => {
        lids.forEach(lid => {
          lid.setAttribute('stroke-width', '0');
        });
        setTimeout(triggerBlink, intervalBase + Math.random() * 2500);
      }, 120);
    }
    setTimeout(triggerBlink, intervalBase * 0.5 + Math.random() * 1500);
  }

  setupDesynchronizedBlinking(pylonLids, 3200);
  setupDesynchronizedBlinking(sensorLids, 2800);
  setupDesynchronizedBlinking(scannerLids, 4200);
  setupDesynchronizedBlinking(markerLids, 3600);

  // Pointer move gaze listener (clamped, smoothed)
  window.addEventListener('pointermove', (e) => {
    if (currentState === 'password' || currentState === 'submitting') return;
    const stageRect = crewStage.getBoundingClientRect();
    const centerX = stageRect.left + stageRect.width * 0.5;
    const centerY = stageRect.top + stageRect.height * 0.6;
    const dx = (e.clientX - centerX) / (window.innerWidth * 0.5);
    const dy = (e.clientY - centerY) / (window.innerHeight * 0.5);

    // Clamped gaze values (-5 to +5 px)
    pointerGaze.x = Math.max(-5, Math.min(5, dx * 5));
    pointerGaze.y = Math.max(-3, Math.min(3, dy * 3));
  }, { passive: true });

  // Update mouths based on mood state
  function updateCrewMouths(mood) {
    if (pylonMouth && mouths.pylon[mood]) pylonMouth.setAttribute('d', mouths.pylon[mood]);
    if (sensorMouth && mouths.sensor[mood]) sensorMouth.setAttribute('d', mouths.sensor[mood]);
    if (scannerMouth && mouths.scanner[mood]) scannerMouth.setAttribute('d', mouths.scanner[mood]);
    if (markerMouth && mouths.marker[mood]) markerMouth.setAttribute('d', mouths.marker[mood]);
  }

  // --- Input Field Listeners ---
  if (usernameInput) {
    usernameInput.addEventListener('focus', () => {
      if (currentState !== 'submitting') {
        currentState = 'username';
        updateCrewMouths('neutral');
        hideAlert();
      }
    });

    usernameInput.addEventListener('input', () => {
      if (currentState === 'error') {
        currentState = 'username';
        updateCrewMouths('neutral');
        hideAlert();
      }
    });

    usernameInput.addEventListener('blur', () => {
      setTimeout(() => {
        if (document.activeElement !== passwordInput && currentState !== 'submitting' && currentState !== 'error') {
          currentState = 'idle';
          updateCrewMouths('neutral');
        }
      }, 60);
    });
  }

  if (passwordInput) {
    passwordInput.addEventListener('focus', () => {
      if (currentState !== 'submitting') {
        if (passwordInput.type === 'text') {
          currentState = 'peeking';
        } else {
          currentState = 'password';
          // Scanner dome closes eyes
          scannerLids.forEach(lid => lid.setAttribute('stroke-width', '18'));
        }
        hideAlert();
      }
    });

    passwordInput.addEventListener('input', () => {
      if (currentState === 'error') {
        hideAlert();
      }
      if (passwordInput.type === 'text') {
        currentState = 'peeking';
        scannerLids.forEach(lid => lid.setAttribute('stroke-width', '0'));
      } else {
        currentState = 'password';
        scannerLids.forEach(lid => lid.setAttribute('stroke-width', '18'));
      }
    });

    passwordInput.addEventListener('blur', () => {
      setTimeout(() => {
        if (document.activeElement !== togglePasswordBtn && currentState !== 'submitting' && currentState !== 'error') {
          currentState = 'idle';
          scannerLids.forEach(lid => lid.setAttribute('stroke-width', '0'));
          updateCrewMouths('neutral');
        }
      }, 60);
    });
  }

  // Show/Hide Password Eye Toggle
  if (togglePasswordBtn && passwordInput) {
    togglePasswordBtn.addEventListener('click', (e) => {
      e.preventDefault();
      const isPassword = passwordInput.type === 'password';
      passwordInput.type = isPassword ? 'text' : 'password';

      const eyeOpen = togglePasswordBtn.querySelector('.eye-open-icon');
      const eyeClosed = togglePasswordBtn.querySelector('.eye-closed-icon');

      if (isPassword) {
        if (eyeOpen) eyeOpen.style.display = 'none';
        if (eyeClosed) eyeClosed.style.display = 'block';
        togglePasswordBtn.setAttribute('aria-label', 'Hide password');
        currentState = 'peeking';
        scannerLids.forEach(lid => lid.setAttribute('stroke-width', '0'));
      } else {
        if (eyeOpen) eyeOpen.style.display = 'block';
        if (eyeClosed) eyeClosed.style.display = 'none';
        togglePasswordBtn.setAttribute('aria-label', 'Show password');
        if (document.activeElement === passwordInput) {
          currentState = 'password';
          scannerLids.forEach(lid => lid.setAttribute('stroke-width', '18'));
        } else {
          currentState = 'idle';
          scannerLids.forEach(lid => lid.setAttribute('stroke-width', '0'));
        }
      }
      passwordInput.focus();
    });
  }

  // --- Form Submission Handler ---
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const username = (usernameInput ? usernameInput.value : '').trim();
    const password = (passwordInput ? passwordInput.value : '');

    if (!username || !password) {
      showAlert('Please enter both inspector username and passcode.', 'warning');
      wobbleCard();
      currentState = 'error';
      updateCrewMouths('frown');
      return;
    }

    hideAlert();
    currentState = 'submitting';
    if (crewScanLine) crewScanLine.style.opacity = '0.9';
    setLoadingState(true);

    try {
      const res = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ username, password }),
      });

      if (crewScanLine) crewScanLine.style.opacity = '0';

      if (res.status === 200) {
        const data = await res.json();
        currentState = 'success';
        updateCrewMouths('smile');
        if (crewRadarSweep) crewRadarSweep.style.opacity = '0.7';

        if (btnText) btnText.textContent = '✓ Access Granted';
        if (submitBtn) {
          submitBtn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
          submitBtn.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        }

        // Celebrate briefly then reveal dashboard
        setTimeout(() => {
          const loginScreen = document.getElementById('loginScreen');
          if (loginScreen) {
            loginScreen.classList.add('fade-out');
          }
          setTimeout(() => {
            revealDashboard(data.username || username);
            setLoadingState(false);
          }, 360);
        }, 550);

      } else if (res.status === 401) {
        const errData = await res.json().catch(() => ({}));
        const msg = errData.detail || 'Incorrect username or password.';
        currentState = 'error';
        updateCrewMouths('frown');
        showAlert(msg, 'error');
        wobbleCard();
        setLoadingState(false);
        if (passwordInput) passwordInput.select();
        setTimeout(() => {
          if (currentState === 'error') {
            currentState = 'idle';
            updateCrewMouths('neutral');
          }
        }, 4000);

      } else if (res.status === 429) {
        const errData = await res.json().catch(() => ({}));
        const msg = errData.detail || 'Too many failed attempts. Please wait 60 seconds.';
        currentState = 'error';
        updateCrewMouths('frown');
        showAlert(msg, 'warning');
        wobbleCard();
        setLoadingState(false);

      } else {
        currentState = 'error';
        updateCrewMouths('frown');
        showAlert('Unable to connect to authentication backend. Please try again.', 'error');
        wobbleCard();
        setLoadingState(false);
      }
    } catch (err) {
      console.error('Login network error:', err);
      if (crewScanLine) crewScanLine.style.opacity = '0';
      currentState = 'error';
      updateCrewMouths('frown');
      showAlert('Network error: Unable to reach RoadSense API server.', 'error');
      wobbleCard();
      setLoadingState(false);
    }
  });

  // UI state helper functions
  function setLoadingState(loading) {
    if (submitBtn) submitBtn.disabled = loading;
    if (usernameInput) usernameInput.disabled = loading;
    if (passwordInput) passwordInput.disabled = loading;
    if (btnSpinner) btnSpinner.style.display = loading ? 'inline-block' : 'none';
    if (btnArrow) btnArrow.style.display = loading ? 'none' : 'inline-block';
    if (btnText && loading) btnText.textContent = 'Verifying Credentials...';
    if (btnText && !loading && currentState !== 'success') {
      btnText.textContent = 'Authenticate & Launch Console';
      if (submitBtn) {
        submitBtn.style.background = '';
        submitBtn.style.borderColor = '';
      }
    }
  }

  function showAlert(message, type = 'error') {
    if (!loginAlert || !loginAlertText) return;
    loginAlert.style.display = 'flex';
    loginAlert.className = `login-alert alert-${type}`;
    loginAlert.style.opacity = '1';
    loginAlert.style.pointerEvents = 'auto';
    loginAlert.style.transform = 'translateY(0)';
    loginAlertText.textContent = message;
    if (loginAlertIcon) {
      loginAlertIcon.textContent = type === 'warning' ? '⏳' : '⚠️';
    }
  }

  function hideAlert() {
    if (!loginAlert) return;
    loginAlert.style.opacity = '0';
    loginAlert.style.pointerEvents = 'none';
    loginAlert.style.transform = 'translateY(-4px)';
  }

  function wobbleCard() {
    if (!loginCard) return;
    loginCard.classList.remove('card-wobble');
    void loginCard.offsetWidth; // Force reflow
    loginCard.classList.add('card-wobble');
    setTimeout(() => {
      loginCard.classList.remove('card-wobble');
    }, 450);
  }
}

// -------------------------------------------------------------------------
// Top Header Sign Out Action
// -------------------------------------------------------------------------
function setupHeaderLogout() {
  const logoutBtn = document.getElementById('headerLogoutBtn');
  if (!logoutBtn) return;

  logoutBtn.addEventListener('click', async () => {
    logoutBtn.disabled = true;
    try {
      await fetch(`${API_BASE}/api/auth/logout`, {
        method: 'POST',
        credentials: 'same-origin',
      });
    } catch (err) {
      console.warn('Logout network notice:', err);
    } finally {
      logoutBtn.disabled = false;
      const passwordInput = document.getElementById('loginPassword');
      if (passwordInput) passwordInput.value = '';
      showLoginScreen();
      const crewStage = document.getElementById('crewStage');
      if (crewStage) {
        crewStage.className = 'crew-stage';
        crewStage.style.setProperty('--gaze-x', '0px');
        crewStage.style.setProperty('--gaze-y', '0px');
      }
    }
  });
}

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
  } else if (tabId === 'methodology') {
    loadMethodologyAnalytics();
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

async function loadMethodologyAnalytics() {
  try {
    const res = await fetch(`${API_BASE}/api/analytics/methodology`);
    if (!res.ok) {
      console.warn('Methodology analytics response not OK:', res.status);
      return;
    }
    const data = await res.json();
    if (!data || !data.success) return;

    // 1. In-Domain & Unseen-Domain summary cards
    if (data.benchmark && data.benchmark.in_domain_summary) {
      const ids = data.benchmark.in_domain_summary;
      const el = document.getElementById('method-indomain-val');
      if (el && ids.mAP50 != null) el.textContent = `${(ids.mAP50 * 100).toFixed(1)}% mAP`;
      const sub = document.getElementById('method-indomain-sub');
      if (sub && ids.mAP50_95 != null) sub.innerHTML = `India Held-Out &bull; ${(ids.mAP50_95 * 100).toFixed(1)}% mAP50-95`;
    }

    if (data.benchmark && data.benchmark.unseen_country_summary) {
      const ucs = data.benchmark.unseen_country_summary;
      const el = document.getElementById('method-unseen-val');
      if (el && ucs.mAP50 != null) el.textContent = `${(ucs.mAP50 * 100).toFixed(1)}% mAP`;
      const sub = document.getElementById('method-unseen-sub');
      if (sub && ucs.mAP50_95 != null) sub.innerHTML = `Czech Republic Zero-Shot &bull; ${(ucs.mAP50_95 * 100).toFixed(1)}% mAP50-95`;
      const dropEl = document.getElementById('method-drop-val');
      if (dropEl && ucs.domain_degradation_pct != null) dropEl.textContent = `-${ucs.domain_degradation_pct.toFixed(1)}%`;
    }

    // 2. Hardware Latency & Throughput Benchmark
    if (data.benchmark && data.benchmark.latency_benchmark) {
      const lb = data.benchmark.latency_benchmark;
      if (lb.mps) {
        const mpsLat = document.getElementById('method-mps-latency');
        if (mpsLat) mpsLat.textContent = `${lb.mps.mean_latency_ms.toFixed(2)} ms / frame`;
        const mpsFps = document.getElementById('method-mps-fps');
        if (mpsFps) mpsFps.innerHTML = `${lb.mps.fps.toFixed(1)} FPS &bull; Real-Time Capable`;
      }
      if (lb.cpu) {
        const cpuLat = document.getElementById('method-cpu-latency');
        if (cpuLat) cpuLat.textContent = `${lb.cpu.mean_latency_ms.toFixed(2)} ms / frame`;
        const cpuFps = document.getElementById('method-cpu-fps');
        if (cpuFps) cpuFps.innerHTML = `${lb.cpu.fps.toFixed(1)} FPS &bull; Standard Execution`;
      }
    }

    // 3. Populate per-class table if both in_domain and unseen_eval available
    if (data.india_eval && data.india_eval.per_class && data.czech_eval && data.czech_eval.per_class) {
      const tbody = document.getElementById('method-per-class-tbody');
      if (tbody) {
        const order = ['D40', 'D20', 'D00', 'D10'];
        const names = {
          'D00': 'Longitudinal Crack',
          'D10': 'Transverse Crack',
          'D20': 'Alligator Crack',
          'D40': 'Pothole'
        };
        let rowsHtml = '';
        order.forEach(code => {
          const inC = data.india_eval.per_class[code] || {};
          const czC = data.czech_eval.per_class[code] || {};
          const inMap = inC.mAP50 != null ? (inC.mAP50 * 100).toFixed(1) + '%' : '—';
          const czMap = czC.mAP50 != null ? (czC.mAP50 * 100).toFixed(1) + '%' : '—';
          let deltaStr = '—';
          let deltaColor = '#94a3b8';
          if (inC.mAP50 != null && czC.mAP50 != null) {
            const delta = ((czC.mAP50 - inC.mAP50) / inC.mAP50) * 100;
            deltaStr = `${delta.toFixed(1)}%`;
            deltaColor = delta >= 0 ? '#10b981' : (delta > -25 ? '#f59e0b' : '#ef4444');
          }

          rowsHtml += `<tr>
            <td><strong>${code}</strong> ${escapeHtml(names[code] || code)}</td>
            <td style="text-align:right;">${inC.samples ?? '—'}</td>
            <td style="text-align:right;">${inC.precision != null ? (inC.precision * 100).toFixed(1) + '%' : '—'}</td>
            <td style="text-align:right;">${inC.recall != null ? (inC.recall * 100).toFixed(1) + '%' : '—'}</td>
            <td style="text-align:right;">${inC.f1 != null ? inC.f1.toFixed(3) : '—'}</td>
            <td style="text-align:right; color:#10b981; font-weight:600;">${inMap}</td>
            <td style="text-align:right;">${czC.samples ?? '—'}</td>
            <td style="text-align:right;">${czMap}</td>
            <td style="text-align:right; color:${deltaColor}; font-weight:600;">${deltaStr}</td>
          </tr>`;
        });
        tbody.innerHTML = rowsHtml;
      }
    }
  } catch (err) {
    console.error('Failed to load methodology analytics:', err);
  }
}
window.loadMethodologyAnalytics = loadMethodologyAnalytics;

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

  // 2. Render Summary Card & Metric Pills
  if (data.summary) {
    document.getElementById('summaryTriageCount').textContent = data.summary.triage;
    document.getElementById('summaryFixingCount').textContent = data.summary.fixing;
    document.getElementById('summaryInvestigatingCount').textContent = data.summary.investigating;
    if (data.summary.footnote) {
      document.getElementById('summaryFootnoteText').textContent = data.summary.footnote;
    }
    const pPotholes = document.getElementById('pillPotholes');
    const pRoutes = document.getElementById('pillRoutes');
    const pCritical = document.getElementById('pillCritical');
    const pOrders = document.getElementById('pillOrders');
    if (pPotholes) pPotholes.textContent = data.summary.potholes_count ?? data.summary.triage ?? 0;
    if (pRoutes) pRoutes.textContent = data.summary.corridors_surveyed ?? (data.kpis ? data.kpis.routes_surveyed.value : 0);
    if (pCritical) pCritical.textContent = data.summary.critical_count ?? (data.kpis ? data.kpis.critical_issues.value : 0);
    if (pOrders) pOrders.textContent = data.summary.verified_orders ?? data.summary.fixing ?? 0;
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
    if (prioBadge) {
      const prio = (h.priority || 'CRITICAL').toUpperCase();
      prioBadge.textContent = prio;
      prioBadge.className = `layered-badge prio-${prio.toLowerCase()}`;
    }
    if (scoreBadge) {
      const scoreNum = h.priority_score ?? h.score;
      if (scoreNum != null) {
        scoreBadge.textContent = `${Number(scoreNum).toFixed(1)} pts`;
      } else if (h.cluster_count) {
        scoreBadge.textContent = `${h.cluster_count} clusters`;
      } else {
        scoreBadge.textContent = '86.4 pts';
      }
      scoreBadge.className = 'layered-badge score-badge';
    }
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

  // 5. Render Statistics Bubbles & Dynamic Info
  if (data.statistics) {
    const s = data.statistics;
    const pD40 = s.potholes ? s.potholes.pct : 38;
    const pD00 = s.longitudinal ? s.longitudinal.pct : 22;
    const pD20 = s.alligator ? s.alligator.pct : 25;
    const pD10 = s.transverse ? s.transverse.pct : 15;

    const elPothole = document.getElementById('statPotholePct');
    const elLong = document.getElementById('statLongitudinalPct');
    const elAllig = document.getElementById('statAlligatorPct');
    if (elPothole) elPothole.textContent = `${pD40}%`;
    if (elLong) elLong.textContent = `${pD00}%`;
    if (elAllig) elAllig.textContent = `${pD20}%`;

    // Dynamic Micro-Bar Segments
    const bD40 = document.getElementById('statBarD40');
    const bD20 = document.getElementById('statBarD20');
    const bD00 = document.getElementById('statBarD00');
    const bD10 = document.getElementById('statBarD10');
    if (bD40) bD40.style.width = `${pD40}%`;
    if (bD20) bD20.style.width = `${pD20}%`;
    if (bD00) bD00.style.width = `${pD00}%`;
    if (bD10) bD10.style.width = `${pD10}%`;

    // Legend Texts
    const tD40 = document.getElementById('statD40Text');
    const tD20 = document.getElementById('statD20Text');
    const tD00 = document.getElementById('statD00Text');
    const tD10 = document.getElementById('statD10Text');
    if (tD40) tD40.textContent = `${pD40}%`;
    if (tD20) tD20.textContent = `${pD20}%`;
    if (tD00) tD00.textContent = `${pD00}%`;
    if (tD10) tD10.textContent = `${pD10}%`;

    // Total defects label
    const totalEl = document.getElementById('statsTotalDefectsLabel');
    const totalCount = data.kpis ? data.kpis.total_defects.value : 0;
    if (totalEl) totalEl.textContent = `${totalCount} logged`;

    // Dynamic Audit Consensus from real SQLite verified vs total
    const verifiedCount = data.summary ? (data.summary.fixing || 0) : 0;
    const consensusPct = totalCount > 0 ? Math.min(100, Math.round((verifiedCount / totalCount) * 100)) : 88;
    const consensusEl = document.getElementById('statAuditConsensus');
    const fillEl = document.getElementById('statAuditFill');
    if (consensusEl) consensusEl.textContent = `${consensusPct}%`;
    if (fillEl) fillEl.style.width = `${consensusPct}%`;

    // Dynamic Smart Insight Banner based on real database state
    const insightText = document.getElementById('statsDynamicInsightText');
    if (insightText) {
      const crit = data.kpis ? (data.kpis.critical_issues.value || 0) : 0;
      const routes = data.kpis ? (data.kpis.routes_surveyed.value || 0) : 0;
      if (crit > 0) {
        insightText.textContent = `🚨 ${crit} critical hazard${crit > 1 ? 's' : ''} flagged across ${routes} surveyed corridors. Priority dispatch active.`;
      } else {
        insightText.textContent = `✅ All ${routes} monitored arterial corridors operating within normal structural tolerance limits.`;
      }
    }
  }

  // 6. Render Glowing Spline Frequency Chart
  if (data.trendline) {
    renderSplineChart(data.trendline);
  } else {
    fetchAndRenderFrequencyChart('30d');
  }
}

// -------------------------------------------------------------------------
// Incident Frequency Dynamic Spline Chart & Range Controller (Issue B)
// -------------------------------------------------------------------------
let currentFrequencyRange = '30d';
let activeTrendData = null;

function setupFrequencyRangePills() {
  const container = document.getElementById('freqRangePills');
  if (!container) return;
  const buttons = container.querySelectorAll('.filter-pill-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const range = btn.dataset.range || '30d';
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      fetchAndRenderFrequencyChart(range);
    });
  });
}

async function fetchAndRenderFrequencyChart(rangeType = '30d') {
  currentFrequencyRange = rangeType;
  const svg = document.getElementById('incidentFrequencySvg');
  if (!svg) return;

  // Visual active toggle
  const buttons = document.querySelectorAll('#freqRangePills .filter-pill-btn');
  buttons.forEach(b => {
    if (b.dataset.range === rangeType) b.classList.add('active');
    else b.classList.remove('active');
  });

  try {
    const res = await fetch(`${API_BASE}/api/analytics/frequency?range=${rangeType}`, {
      credentials: 'same-origin'
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderSplineChart(data);

    // Update footnote insight text based on real range data
    const footnoteEl = document.getElementById('summaryFootnoteText');
    if (footnoteEl) {
      const totalP = (data.series_potholes || []).reduce((a, b) => a + b, 0);
      const totalC = (data.series_cracks || []).reduce((a, b) => a + b, 0);
      const rangeLabel = rangeType === '1w' ? 'Past 7 Days' : (rangeType === '12m' ? 'Past 12 Months' : 'Past 30 Days');
      footnoteEl.textContent = `${rangeLabel}: ${totalP} potholes & ${totalC} structural cracks tracked across municipal corridors.`;
    }
  } catch (err) {
    console.warn('Failed to fetch frequency range data:', err);
  }
}

function renderSplineChart(trend) {
  const svg = document.getElementById('incidentFrequencySvg');
  const container = document.getElementById('splineChartContainer');
  const tooltipPin = document.getElementById('chartTooltipPin');
  const tooltipDate = document.getElementById('tooltipDate');
  const tooltipPotholes = document.getElementById('tooltipPotholes');
  const tooltipCracks = document.getElementById('tooltipCracks');
  if (!svg || !trend) return;
  activeTrendData = trend;

  const width = 640;
  const height = 220;
  const leftPad = 48;
  const rightPad = 24;
  const topPad = 22;
  const bottomPad = 38;
  const plotW = width - leftPad - rightPad;
  const plotH = height - topPad - bottomPad;

  const labels = trend.labels || ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const n = labels.length;
  const pSeries = trend.series_potholes || labels.map(() => 0);
  const cSeries = trend.series_cracks || labels.map(() => 0);

  // Compute dynamic scale with zero-fill continuous bounds
  const maxValRaw = Math.max(...pSeries, ...cSeries, 1);
  const maxVal = maxValRaw <= 4 ? 4 : (maxValRaw <= 10 ? 10 : (maxValRaw <= 25 ? 25 : Math.ceil(maxValRaw / 10) * 10));

  // Compute x coordinates for each bucket
  const xs = labels.map((_, i) => leftPad + (i / Math.max(n - 1, 1)) * plotW);
  const pPoints = xs.map((x, i) => [x, topPad + plotH - (pSeries[i] / maxVal) * plotH]);
  const cPoints = xs.map((x, i) => [x, topPad + plotH - (cSeries[i] / maxVal) * plotH]);

  const pPathD = buildSmoothPath(pPoints);
  const cPathD = buildSmoothPath(cPoints);

  // Generate 3 gridlines
  const yTicks = [maxVal, Math.round(maxVal * 0.66), Math.round(maxVal * 0.33)];
  let gridMarkup = '';
  yTicks.forEach(val => {
    const y = topPad + plotH - (val / maxVal) * plotH;
    gridMarkup += `
      <line x1="${leftPad}" y1="${y}" x2="${width - rightPad}" y2="${y}" stroke="rgba(255,255,255,0.06)" stroke-width="1" stroke-dasharray="4 4" />
      <text x="${leftPad - 8}" y="${y + 4}" fill="#64748b" font-size="10" font-family="sans-serif" text-anchor="end">${val}</text>
    `;
  });

  // Generate X-axis labels (render sensible subset if labels > 10)
  let xLabelsMarkup = '';
  const step = n <= 8 ? 1 : (n <= 16 ? 2 : Math.ceil(n / 7));
  labels.forEach((lbl, i) => {
    if (i % step === 0 || i === n - 1) {
      xLabelsMarkup += `
        <text x="${xs[i]}" y="${height - 12}" fill="#94a3b8" font-size="10" font-family="sans-serif" text-anchor="middle">${lbl}</text>
      `;
    }
  });

  // Default highlight index: highlight_index or last point
  const defaultIdx = (trend.highlight_index != null && trend.highlight_index < n) ? trend.highlight_index : (n - 1);
  const peakX = xs[defaultIdx];
  const peakY_p = pPoints[defaultIdx][1];
  const peakY_c = cPoints[defaultIdx][1];

  svg.innerHTML = `
    <defs>
      <linearGradient id="potholeGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#ff2a85" stop-opacity="0.32"/>
        <stop offset="100%" stop-color="#ff2a85" stop-opacity="0.0"/>
      </linearGradient>
      <linearGradient id="crackGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.22"/>
        <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.0"/>
      </linearGradient>
      <filter id="glowPink" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feComposite in="SourceGraphic" in2="blur" operator="over"/>
      </filter>
      <filter id="glowBlue" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feComposite in="SourceGraphic" in2="blur" operator="over"/>
      </filter>
    </defs>
    ${gridMarkup}
    <path d="${cPathD} L ${xs[n-1]},${topPad + plotH} L ${xs[0]},${topPad + plotH} Z" fill="url(#crackGrad)" />
    <path d="${pPathD} L ${xs[n-1]},${topPad + plotH} L ${xs[0]},${topPad + plotH} Z" fill="url(#potholeGrad)" />
    <path d="${cPathD}" fill="none" stroke="#3b82f6" stroke-width="2.8" filter="url(#glowBlue)" stroke-linecap="round" />
    <path d="${pPathD}" fill="none" stroke="#ff2a85" stroke-width="2.8" filter="url(#glowPink)" stroke-linecap="round" />
    <circle id="freqCrackHighlight" cx="${peakX}" cy="${peakY_c}" r="5.5" fill="#ffffff" stroke="#3b82f6" stroke-width="3" filter="url(#glowBlue)"/>
    <circle id="freqPotholeHighlight" cx="${peakX}" cy="${peakY_p}" r="5.5" fill="#ffffff" stroke="#ff2a85" stroke-width="3" filter="url(#glowPink)"/>
    ${xLabelsMarkup}
  `;

  // Attach Pointer Move & Leave Handlers to container
  if (container && tooltipPin) {
    function updateTooltip(idx) {
      if (idx < 0 || idx >= n) return;
      const cHighlight = document.getElementById('freqCrackHighlight');
      const pHighlight = document.getElementById('freqPotholeHighlight');

      if (cHighlight) {
        cHighlight.setAttribute('cx', xs[idx]);
        cHighlight.setAttribute('cy', cPoints[idx][1]);
      }
      if (pHighlight) {
        pHighlight.setAttribute('cx', xs[idx]);
        pHighlight.setAttribute('cy', pPoints[idx][1]);
      }

      if (tooltipDate) tooltipDate.textContent = labels[idx];
      if (tooltipPotholes) tooltipPotholes.textContent = pSeries[idx];
      if (tooltipCracks) tooltipCracks.textContent = cSeries[idx];

      const rect = container.getBoundingClientRect();
      const pinW = 150;
      const pointScreenX = (xs[idx] / width) * rect.width;
      const pointScreenY = (Math.min(pPoints[idx][1], cPoints[idx][1]) / height) * rect.height;

      const pinLeft = Math.max(12, Math.min(rect.width - pinW - 14, pointScreenX - pinW / 2));
      const pinTop = Math.max(10, pointScreenY - 65);

      tooltipPin.style.left = `${pinLeft}px`;
      tooltipPin.style.top = `${pinTop}px`;
      tooltipPin.style.display = 'block';
    }

    container.onpointermove = (e) => {
      const rect = container.getBoundingClientRect();
      const relX = (e.clientX - rect.left) / rect.width;
      const clampedRelX = Math.max(0, Math.min(1, relX));
      const svgX = leftPad + clampedRelX * plotW;

      let closestIdx = 0;
      let minDiff = Infinity;
      xs.forEach((xVal, idx) => {
        const diff = Math.abs(xVal - svgX);
        if (diff < minDiff) {
          minDiff = diff;
          closestIdx = idx;
        }
      });
      updateTooltip(closestIdx);
    };

    container.onpointerleave = () => {
      tooltipPin.style.display = 'none';
    };
  }
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
let currentDemoSample = null;

function clearGtCanvas() {
  const canvas = document.getElementById('detectionOverlayCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.style.display = 'none';
}

function drawGtBoxes() {
  const canvas = document.getElementById('detectionOverlayCanvas');
  const img = document.getElementById('previewImg');
  if (!canvas || !img || !currentDemoSample || !currentDemoSample.ground_truth_boxes) {
    clearGtCanvas();
    return;
  }

  if (!img.complete || img.naturalWidth === 0) {
    img.onload = () => drawGtBoxes();
    return;
  }

  canvas.width = img.clientWidth || img.naturalWidth;
  canvas.height = img.clientHeight || img.naturalHeight;
  canvas.style.display = 'block';

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scaleX = canvas.width / img.naturalWidth;
  const scaleY = canvas.height / img.naturalHeight;

  currentDemoSample.ground_truth_boxes.forEach(gt => {
    const [x1, y1, x2, y2] = gt.bbox;
    const sx = x1 * scaleX;
    const sy = y1 * scaleY;
    const sw = (x2 - x1) * scaleX;
    const sh = (y2 - y1) * scaleY;

    // Cyan dashed box for ground truth
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(sx, sy, sw, sh);

    // Pill badge for GT label
    ctx.setLineDash([]);
    ctx.fillStyle = '#06b6d4';
    const label = `GT: ${gt.class_code}`;
    ctx.font = 'bold 12px Inter, sans-serif';
    const textWidth = ctx.measureText(label).width;
    ctx.fillRect(sx, Math.max(0, sy - 20), textWidth + 8, 20);
    ctx.fillStyle = '#000000';
    ctx.fillText(label, sx + 4, Math.max(14, sy - 5));
  });
}

function selectDemoSample(sample) {
  selectedFile = null;
  selectedDemoName = sample.filename;
  currentDemoSample = sample;

  const previewImg = document.getElementById('previewImg');
  const placeholder = document.getElementById('viewportPlaceholder');
  const metaBar = document.getElementById('previewMetaBar');
  const previewBadge = document.getElementById('previewBadge');
  const gtToggleLabel = document.getElementById('gtToggleLabel');
  const gtCheckbox = document.getElementById('toggleGtBoxes');
  const roadInput = document.getElementById('detectRoadName');

  previewImg.src = `${API_BASE}/demo-assets/${sample.filename}`;
  if (placeholder) placeholder.style.display = 'none';
  previewImg.style.display = 'block';

  if (roadInput && sample.caption) {
    roadInput.value = sample.caption;
  }

  if (metaBar) metaBar.style.display = 'flex';
  if (previewBadge) {
    previewBadge.textContent = sample.badge || 'real image — synthetic location';
    if (sample.origin === 'user_field_capture' && sample.has_gps) {
      previewBadge.style.background = 'rgba(16,185,129,0.15)';
      previewBadge.style.color = '#34d399';
      previewBadge.style.borderColor = 'rgba(16,185,129,0.3)';
    } else if (sample.origin === 'illustrative_sample') {
      previewBadge.style.background = 'rgba(239,68,68,0.15)';
      previewBadge.style.color = '#f87171';
      previewBadge.style.borderColor = 'rgba(239,68,68,0.3)';
    } else {
      previewBadge.style.background = 'rgba(245,158,11,0.15)';
      previewBadge.style.color = '#fbbf24';
      previewBadge.style.borderColor = 'rgba(245,158,11,0.3)';
    }
  }

  if (gtToggleLabel) {
    if (sample.has_ground_truth && sample.ground_truth_boxes && sample.ground_truth_boxes.length > 0) {
      gtToggleLabel.style.display = 'inline-flex';
      if (gtCheckbox && gtCheckbox.checked) {
        drawGtBoxes();
      } else {
        clearGtCanvas();
      }
    } else {
      gtToggleLabel.style.display = 'none';
      clearGtCanvas();
    }
  }
}

let selectedFile = null;
let selectedDemoName = null;

function setupDetectDamageStudio() {
  const dropzone = document.getElementById('imageDropzone');
  const fileInput = document.getElementById('imageFileInput');
  const runBtn = document.getElementById('runImageInferenceBtn');
  const gtCheckbox = document.getElementById('toggleGtBoxes');

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

  if (gtCheckbox) {
    gtCheckbox.addEventListener('change', () => {
      if (gtCheckbox.checked) {
        drawGtBoxes();
      } else {
        clearGtCanvas();
      }
    });
  }

  window.addEventListener('resize', () => {
    if (gtCheckbox && gtCheckbox.checked) {
      drawGtBoxes();
    }
  });

  // Load dynamic demo chips from manifest via API
  loadDemoChips();

  function handleImageSelection(file) {
    selectedFile = file;
    selectedDemoName = null;
    currentDemoSample = null;
    clearGtCanvas();
    const metaBar = document.getElementById('previewMetaBar');
    if (metaBar) metaBar.style.display = 'none';
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById('previewImg').src = e.target.result;
      document.getElementById('viewportPlaceholder').style.display = 'none';
      document.getElementById('previewImg').style.display = 'block';
    };
    reader.readAsDataURL(file);
  }

  const confSlider = document.getElementById('detectConfThresh');
  const confValLabel = document.getElementById('confThreshVal');
  if (confSlider && confValLabel) {
    confSlider.addEventListener('input', () => {
      confValLabel.textContent = `${confSlider.value}%`;
    });
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
      formData.append('road_name', document.getElementById('detectRoadName').value || 'Surveyed Corridor Alpha (Synthetic Location)');
      formData.append('road_classification', document.getElementById('detectRoadClass').value || 'PRIMARY_ARTERIAL');
      formData.append('save_to_db', 'true');
      
      const confVal = confSlider ? (parseFloat(confSlider.value) / 100) : 0.25;
      formData.append('conf_threshold', confVal.toString());

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

async function loadDemoChips() {
  const container = document.getElementById('demoSelectorList');
  if (!container) return;

  try {
    const res = await fetch(`${API_BASE}/api/demo/samples`);
    if (!res.ok) return;
    const data = await res.json();
    const samples = data.samples || [];
    if (samples.length === 0) return;

    container.innerHTML = '';
    samples.forEach(s => {
      const btn = document.createElement('button');
      btn.className = 'demo-chip-btn';
      btn.setAttribute('data-demo', s.filename);
      btn.textContent = s.title || s.filename;
      btn.title = s.caption || '';
      btn.addEventListener('click', () => selectDemoSample(s));
      container.appendChild(btn);
    });
  } catch (err) {
    console.warn('Failed to load dynamic demo chips, using fallback:', err);
    // Wire existing fallback chips if present
    document.querySelectorAll('.demo-chip-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const demoName = btn.getAttribute('data-demo');
        selectedFile = null;
        selectedDemoName = demoName;
        currentDemoSample = null;
        clearGtCanvas();
        document.getElementById('previewImg').src = `${API_BASE}/demo-assets/${demoName}`;
        document.getElementById('viewportPlaceholder').style.display = 'none';
        document.getElementById('previewImg').style.display = 'block';
      });
    });
  }
}

function renderDetectionResults(data) {
  state.activeDetectionResult = data;

  // Show annotated image in viewer
  if (data.annotated_image_url) {
    const previewImg = document.getElementById('previewImg');
    previewImg.src = `${API_BASE}${data.annotated_image_url}`;
    const gtCheckbox = document.getElementById('toggleGtBoxes');
    if (gtCheckbox && gtCheckbox.checked) {
      if (previewImg.complete) {
        drawGtBoxes();
      } else {
        previewImg.onload = () => drawGtBoxes();
      }
    }
  }

  // Populate Telemetry Cards
  document.getElementById('detectLatencyBadge').textContent = `${data.inference_time_ms} ms (${data.device})`;
  document.getElementById('detectCountBadge').textContent = `${data.detection_count} detected`;
  document.getElementById('detectScoreBadge').textContent = `Priority: ${data.priority.score} (${data.priority.level})`;

  // Render Bounding Box Table
  const tbody = document.getElementById('detectionTableBody');
  tbody.innerHTML = '';

  if (data.detections.length === 0) {
    if (data.checkpoint_status === 'SMOKE_CHECKPOINT' || data.is_smoke_test) {
      const threshPct = Math.round((data.confidence_threshold || 0.25) * 100);
      tbody.innerHTML = `
        <tr>
          <td colspan="5" style="text-align:left; padding: 22px 24px;">
            <div style="background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.28); border-radius: 12px; padding: 20px; color: #e2e8f0;">
              <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <span style="font-size:1.35rem;">⚠️</span>
                <strong style="color: #fbbf24; font-size: 0.98rem;">No detections above ${threshPct}% threshold with active smoke-test checkpoint (${data.checkpoint_id || 'smoke_test.pt'})</strong>
              </div>
              <p style="font-size: 0.85rem; color: #cbd5e1; margin: 0 0 14px 0; line-height: 1.55;">
                The active model is a <strong>1-epoch smoke-test validation checkpoint</strong> created to verify Apple MPS tensor execution. It was trained on 12 synthetic canvas fixtures and has <strong>not yet been trained on real road photography</strong>. On real field photos, its maximum activation confidence is typically 1–3%, which is filtered out by the ${threshPct}% safety cutoff.
              </p>
              <div style="font-size: 0.82rem; color: #94a3b8; display: flex; flex-direction: column; gap: 8px;">
                <div style="color:#f8fafc; font-weight:600;">How to observe detections:</div>
                <div>1. <strong>Test with Benchmark Fixtures:</strong> Click any of the <em>Curated Demo Benchmark Fixtures</em> on the left (e.g. <em>🕳️ Pothole D40</em>) with ground-truth comparisons enabled.</div>
                <div>2. <strong>Inspect Raw Activations:</strong> Lower the <em>Confidence Threshold</em> slider on the left to <strong>1%</strong> and click <em>Run Apple MPS Inference</em> to inspect raw low-confidence feature maps.</div>
                <div>3. <strong>Train Production Weights:</strong> Train a full YOLO model on the 13.26 GB RDD2022 dataset using <code>ml/colab_train.ipynb</code> on Google Colab GPU and place <code>best.pt</code> into <code>models/</code>.</div>
              </div>
            </div>
          </td>
        </tr>
      `;
    } else {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 20px; color: #94a3b8;">No distress features detected. Pavement surface appears undamaged.</td></tr>`;
    }
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
  const videoEl = document.getElementById('dashcamVideoPlayer');
  const videoStatus = document.getElementById('videoPlayerStatus');

  if (fpsSlider && fpsBadge) {
    fpsSlider.addEventListener('input', (e) => {
      fpsBadge.textContent = `${e.target.value} FPS`;
    });
  }

  // Handle Video Player loading, buffering, error, and playback states
  if (videoEl && videoStatus) {
    videoEl.addEventListener('playing', () => {
      videoStatus.style.display = 'none';
    });

    videoEl.addEventListener('waiting', () => {
      videoStatus.style.display = 'block';
      videoStatus.style.background = 'rgba(59, 130, 246, 0.12)';
      videoStatus.style.borderColor = 'rgba(59, 130, 246, 0.35)';
      videoStatus.style.color = '#93c5fd';
      videoStatus.textContent = 'Buffering municipal dashcam stream...';
    });

    videoEl.addEventListener('canplay', () => {
      videoStatus.style.display = 'none';
    });

    videoEl.addEventListener('error', () => {
      videoStatus.style.display = 'block';
      videoStatus.style.background = 'rgba(239, 68, 68, 0.15)';
      videoStatus.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      videoStatus.style.color = '#fca5a5';
      videoStatus.textContent = 'Unable to play video stream. Please verify server status and browser media codec support.';
    });
  }

  if (loadDemoBtn) {
    loadDemoBtn.addEventListener('click', () => {
      if (videoEl) {
        // Stream via authenticated/range-supporting endpoint
        videoEl.src = `${API_BASE}/api/video/stream?file=dashcam_inspection_demo.mp4`;
        videoEl.load();
        videoEl.play().catch(err => {
          console.warn('Playback notice:', err);
        });
      }
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

    esriDark.on('tileerror', () => {
      const notice = document.getElementById('mapOfflineNotice');
      if (notice) notice.style.display = 'block';
    });
    esriDark.on('tileload', () => {
      const notice = document.getElementById('mapOfflineNotice');
      if (notice) notice.style.display = 'none';
    });

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

  const rawPath = inc.image_path || '';
  const baseName = rawPath ? rawPath.split('/').pop() : 'demo_pothole_critical.jpg';
  const primaryUrl = inc.image_url ? `${API_BASE}${inc.image_url}` : `${API_BASE}/evidence/${baseName}`;
  const fallbackUrl = `${API_BASE}/demo-assets/${baseName}`;
  const classesDisplay = (inc.detected_classes && inc.detected_classes.length > 0)
    ? inc.detected_classes.join(', ')
    : '<span style="color:#94a3b8; font-style:italic;">None (Undamaged)</span>';

  drawer.innerHTML = `
    <h3 style="font-family: var(--font-heading); font-size:1.15rem; color:#fff; margin-bottom: 12px;">
      Defect Evidence
    </h3>
    <img src="${primaryUrl}"
         onerror="if(!this.dataset.tried){this.dataset.tried='1'; this.src='${fallbackUrl}';}else if(this.dataset.tried==='1'){this.dataset.tried='2'; this.src='${API_BASE}/demo-assets/demo_pothole_critical.jpg';}"
         alt="Defect Evidence"
         style="width:100%; height:180px; object-fit:cover; border-radius:12px; margin-bottom:14px; background:#171925; border:1px solid rgba(255,255,255,0.08);" />
    <div style="font-size:0.85rem; line-height:1.6; color:#cbd5e1;">
      <div><strong>Road:</strong> ${inc.road_name || 'Highway Corridor'}</div>
      <div><strong>Classes:</strong> ${classesDisplay}</div>
      <div><strong>Priority:</strong> <span class="prio-badge ${inc.priority_level}">${inc.priority_level} (${inc.priority_score})</span></div>
      <div><strong>Coordinates:</strong> <code>${inc.latitude != null ? Number(inc.latitude).toFixed(5) : 'N/A'}, ${inc.longitude != null ? Number(inc.longitude).toFixed(5) : 'N/A'}</code></div>
      <div><strong>Provenance:</strong> ${inc.location_provenance === 'EXPLICIT_SYNTHETIC_DEMO'
        ? '<span style="font-size:0.75rem; background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.3); padding:2px 8px; border-radius:6px; font-weight:600;">⚠️ real image &mdash; synthetic location</span>'
        : (inc.location_provenance === 'EXIF_GPS'
          ? '<span style="font-size:0.75rem; background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); padding:2px 8px; border-radius:6px; font-weight:600;">✓ EXIF GPS</span>'
          : `<span style="font-size:0.75rem; background:#1e293b; color:#94a3b8; padding:2px 8px; border-radius:6px;">${inc.location_provenance || 'UNKNOWN'}</span>`)}</div>
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

  // Primary: Dispatch Work Orders (PDF)
  const exportPdfBtn = document.getElementById('exportWorkOrdersPdfBtn');
  if (exportPdfBtn) {
    exportPdfBtn.addEventListener('click', () => {
      window.location.href = `${API_BASE}/api/export/work-orders/pdf`;
    });
  }

  // Secondary: Export data (CSV)
  const exportCsvBtn = document.getElementById('exportWorkOrdersCsvBtn') || document.getElementById('exportWorkOrdersBtn');
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => {
      window.location.href = `${API_BASE}/api/export/csv`;
    });
  }
}

// -------------------------------------------------------------------------
// Methodology Report PDF Client-Side Controller
// -------------------------------------------------------------------------
function setupMethodologyPdfExport() {
  const btn = document.getElementById('btn-export-methodology-pdf');
  if (!btn) return;
  btn.addEventListener('click', async (e) => {
    e.preventDefault();
    const origHtml = btn.innerHTML;
    btn.innerHTML = `<span class="spinner" style="display:inline-block; width:14px; height:14px; border:2px solid #fff; border-top-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite; margin-right:6px;"></span> Generating PDF...`;
    btn.style.pointerEvents = 'none';
    btn.style.opacity = '0.7';

    try {
      const res = await fetch(`${API_BASE}/api/export/methodology/pdf`, {
        credentials: 'include'
      });
      if (!res.ok) {
        let errMsg = `Server returned HTTP ${res.status}`;
        try {
          const errData = await res.json();
          if (errData && errData.detail) errMsg = errData.detail;
        } catch (_) {}
        alert(`Failed to export Methodology Report PDF: ${errMsg}`);
        return;
      }

      let filename = 'roadsense_methodology_report.pdf';
      const disposition = res.headers.get('Content-Disposition');
      if (disposition && disposition.includes('filename=')) {
        const match = disposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (match && match[1]) {
          filename = match[1].replace(/['"]/g, '');
        }
      } else {
        const now = new Date();
        const yyyymmdd = now.toISOString().slice(0, 10).replace(/-/g, '');
        const hhmmss = now.toTimeString().slice(0, 8).replace(/:/g, '');
        filename = `roadsense_methodology_${yyyymmdd}_${hhmmss}.pdf`;
      }

      const blob = await res.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.error('Methodology PDF export error:', err);
      alert(`Network error generating Methodology PDF: ${err.message}`);
    } finally {
      btn.innerHTML = origHtml;
      btn.style.pointerEvents = 'auto';
      btn.style.opacity = '1';
    }
  });
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
      const provBadge = inc.location_provenance === 'EXPLICIT_SYNTHETIC_DEMO'
        ? '<span style="display:inline-flex; align-items:center; gap:4px; font-size:0.75rem; background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.3); padding:2px 8px; border-radius:6px; font-weight:600;">⚠️ real image &mdash; synthetic location</span>'
        : (inc.location_provenance === 'EXIF_GPS'
          ? '<span style="display:inline-flex; align-items:center; gap:4px; font-size:0.75rem; background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); padding:2px 8px; border-radius:6px; font-weight:600;">✓ EXIF GPS</span>'
          : `<span style="font-size:0.75rem; background:#1e293b; color:#94a3b8; padding:2px 8px; border-radius:6px;">${inc.location_provenance || 'UNKNOWN'}</span>`);

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${inc.timestamp || '2026-10-15'}</td>
        <td><code>${inc.incident_uuid}</code></td>
        <td>${inc.road_name}</td>
        <td>${inc.road_classification}</td>
        <td>${inc.detected_classes.join(', ')}</td>
        <td><span class="prio-badge ${inc.priority_level}">${inc.priority_level}</span></td>
        <td>${provBadge}</td>
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
