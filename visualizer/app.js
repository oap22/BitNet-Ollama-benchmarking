const presets = {
  Origin: {
    centerX: -0.5,
    centerY: 0.0,
    zoom: 1.1,
    iterations: 140,
    palette: "solaris",
    juliaX: -0.79,
    juliaY: 0.15,
  },
  "Seahorse Valley": {
    centerX: -0.743643887037151,
    centerY: 0.13182590420533,
    zoom: 360.0,
    iterations: 256,
    palette: "aurora",
    juliaX: -0.7435,
    juliaY: 0.1314,
  },
  "Elephant Valley": {
    centerX: 0.282,
    centerY: 0.008,
    zoom: 18.0,
    iterations: 220,
    palette: "eclipse",
    juliaX: -0.4,
    juliaY: 0.6,
  },
  "Nebula Fold": {
    centerX: -1.25066,
    centerY: 0.02012,
    zoom: 72.0,
    iterations: 244,
    palette: "tide",
    juliaX: 0.285,
    juliaY: 0.013,
  },
};

const paletteIndices = {
  solaris: 0,
  aurora: 1,
  eclipse: 2,
  tide: 3,
};

const state = {
  mode: "mandelbrot",
  centerX: presets["Seahorse Valley"].centerX,
  centerY: presets["Seahorse Valley"].centerY,
  zoom: presets["Seahorse Valley"].zoom,
  iterations: presets["Seahorse Valley"].iterations,
  palette: presets["Seahorse Valley"].palette,
  juliaX: presets["Seahorse Valley"].juliaX,
  juliaY: presets["Seahorse Valley"].juliaY,
  juliaLocked: false,
  ambientMotion: true,
  activePreset: "Seahorse Valley",
};

const interaction = {
  hoverX: state.centerX,
  hoverY: state.centerY,
  mouseDrag: null,
  mouseSuppressClick: false,
  clickTimer: null,
  touchGesture: null,
  frozenTime: 0,
  lastFrameSeconds: 0,
};

const ui = {
  body: document.body,
  mainCanvas: document.querySelector("#fractal-canvas"),
  juliaCanvas: document.querySelector("#julia-canvas"),
  fallback: document.querySelector("#webgl-fallback"),
  controlPanel: document.querySelector("#control-panel"),
  sheetToggle: document.querySelector("#sheet-toggle"),
  presetSelect: document.querySelector("#preset-select"),
  modeSelect: document.querySelector("#mode-select"),
  paletteSelect: document.querySelector("#palette-select"),
  iterationsRange: document.querySelector("#iterations-range"),
  iterationsOutput: document.querySelector("#iterations-output"),
  ambientToggle: document.querySelector("#ambient-toggle"),
  juliaLockToggle: document.querySelector("#julia-lock-toggle"),
  seedLockButton: document.querySelector("#seed-lock-button"),
  resetButton: document.querySelector("#reset-button"),
  modeBadge: document.querySelector("#mode-badge"),
  overlayCaption: document.querySelector("#overlay-caption"),
  modeHint: document.querySelector("#mode-hint"),
  modeHintChip: document.querySelector("#mode-hint-chip"),
  seedReadout: document.querySelector("#seed-readout"),
  seedStatus: document.querySelector("#seed-status"),
  zoomReadout: document.querySelector("#zoom-readout"),
  coordinateReadout: document.querySelector("#coordinate-readout"),
  telemetryMode: document.querySelector("#telemetry-mode"),
  telemetryIterations: document.querySelector("#telemetry-iterations"),
};

const vertexShaderSource = `
attribute vec2 a_position;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const fragmentShaderSource = `
precision highp float;

uniform vec2 u_resolution;
uniform vec2 u_center;
uniform float u_zoom;
uniform float u_iterations;
uniform float u_mode;
uniform vec2 u_juliaC;
uniform float u_palette;
uniform float u_time;

vec3 cosinePalette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
  return a + b * cos(6.28318530718 * (c * t + d));
}

vec3 paletteColor(float t, float palette, float timeJitter) {
  float shifted = clamp(t, 0.0, 1.0) + 0.035 * sin(timeJitter + t * 14.0);

  if (palette < 0.5) {
    return cosinePalette(
      shifted,
      vec3(0.52, 0.42, 0.34),
      vec3(0.42, 0.28, 0.18),
      vec3(1.0, 1.0, 1.0),
      vec3(0.08, 0.16, 0.22)
    );
  }

  if (palette < 1.5) {
    return cosinePalette(
      shifted,
      vec3(0.35, 0.48, 0.6),
      vec3(0.25, 0.25, 0.3),
      vec3(1.0, 1.0, 1.0),
      vec3(0.72, 0.35, 0.18)
    );
  }

  if (palette < 2.5) {
    return cosinePalette(
      shifted,
      vec3(0.18, 0.18, 0.26),
      vec3(0.55, 0.28, 0.22),
      vec3(1.0, 1.0, 1.0),
      vec3(0.82, 0.11, 0.05)
    );
  }

  return cosinePalette(
    shifted,
    vec3(0.32, 0.44, 0.52),
    vec3(0.26, 0.18, 0.16),
    vec3(1.0, 1.0, 1.0),
    vec3(0.54, 0.24, 0.12)
  );
}

vec2 complexSquare(vec2 z) {
  return vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
  float scale = 3.1 / max(u_zoom, 0.0001);

  vec2 c = vec2(0.0);
  vec2 z = vec2(0.0);

  if (u_mode < 0.5) {
    c = u_center + uv * scale;
    z = vec2(0.0);
  } else {
    c = u_juliaC;
    z = u_center + uv * scale;
  }

  float maxIter = max(u_iterations, 1.0);
  float trap = 10.0;
  float iter = 0.0;
  bool escaped = false;

  for (int i = 0; i < 1200; i++) {
    if (float(i) >= maxIter) {
      break;
    }

    z = complexSquare(z) + c;
    trap = min(trap, dot(z, z));

    if (dot(z, z) > 64.0) {
      iter = float(i) + 1.0;
      escaped = true;
      break;
    }
  }

  if (!escaped) {
    float glow = exp(-7.0 * trap);
    float pulse = 0.5 + 0.5 * sin(u_time * 0.15 + trap * 110.0);
    vec3 interior = mix(vec3(0.015, 0.022, 0.05), vec3(0.06, 0.1, 0.18), pulse);
    interior += glow * vec3(0.28, 0.21, 0.14);
    gl_FragColor = vec4(interior, 1.0);
    return;
  }

  float smoothIter = iter - log2(max(log2(length(z)), 0.0001));
  float t = smoothIter / maxIter;
  float shimmer = u_time * 0.18 + length(uv) * 8.0;
  vec3 color = paletteColor(t, u_palette, shimmer);
  float fringe = 0.035 * sin(34.0 * t - u_time * 0.25 + atan(uv.y, uv.x) * 3.0);
  color += fringe * vec3(0.7, 0.45, 0.3);

  float vignette = smoothstep(1.55, 0.18, length(uv));
  color *= vignette;
  color += vec3(0.01, 0.012, 0.028);

  gl_FragColor = vec4(color, 1.0);
}
`;

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const error = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(error || "Shader compilation failed.");
  }

  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const error = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(error || "Program link failed.");
  }

  return program;
}

function resizeCanvasToDisplaySize(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(canvas.clientWidth * dpr);
  const height = Math.round(canvas.clientHeight * dpr);

  if (!width || !height) {
    return false;
  }

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    return true;
  }

  return false;
}

function createRenderer(canvas) {
  const gl = canvas.getContext("webgl", {
    antialias: false,
    preserveDrawingBuffer: false,
    powerPreference: "high-performance",
  });

  if (!gl) {
    return null;
  }

  const program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
    gl.STATIC_DRAW,
  );

  const positionLocation = gl.getAttribLocation(program, "a_position");
  gl.useProgram(program);
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

  const uniforms = {
    resolution: gl.getUniformLocation(program, "u_resolution"),
    center: gl.getUniformLocation(program, "u_center"),
    zoom: gl.getUniformLocation(program, "u_zoom"),
    iterations: gl.getUniformLocation(program, "u_iterations"),
    mode: gl.getUniformLocation(program, "u_mode"),
    juliaC: gl.getUniformLocation(program, "u_juliaC"),
    palette: gl.getUniformLocation(program, "u_palette"),
    time: gl.getUniformLocation(program, "u_time"),
  };

  return {
    canvas,
    gl,
    uniforms,
    render(renderState) {
      resizeCanvasToDisplaySize(canvas);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(program);
      gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
      gl.uniform2f(uniforms.center, renderState.centerX, renderState.centerY);
      gl.uniform1f(uniforms.zoom, renderState.zoom);
      gl.uniform1f(uniforms.iterations, renderState.iterations);
      gl.uniform1f(uniforms.mode, renderState.mode === "julia" ? 1 : 0);
      gl.uniform2f(uniforms.juliaC, renderState.juliaX, renderState.juliaY);
      gl.uniform1f(uniforms.palette, paletteIndices[renderState.palette] ?? 0);
      gl.uniform1f(uniforms.time, renderState.time);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    },
  };
}

let mainRenderer = null;
let juliaRenderer = null;

function formatFloat(value) {
  const digits = state.zoom > 90 ? 7 : state.zoom > 12 ? 6 : 4;
  return Number(value).toFixed(digits);
}

function formatComplex(x, y) {
  const sign = y >= 0 ? "+" : "-";
  return `${formatFloat(x)} ${sign} ${formatFloat(Math.abs(y))}i`;
}

function formatZoom(zoom) {
  if (zoom >= 100) {
    return `x${zoom.toFixed(0)}`;
  }

  if (zoom >= 10) {
    return `x${zoom.toFixed(1)}`;
  }

  return `x${zoom.toFixed(2)}`;
}

function updateSeedReadout() {
  ui.seedReadout.textContent = formatComplex(state.juliaX, state.juliaY);
  ui.seedStatus.textContent = state.juliaLocked ? "Locked to seed" : "Tracking cursor";
}

function updateTelemetry() {
  ui.zoomReadout.textContent = formatZoom(state.zoom);
  ui.coordinateReadout.textContent = formatComplex(interaction.hoverX, interaction.hoverY);
  ui.telemetryMode.textContent = state.mode === "julia" ? "Julia" : "Mandelbrot";
  ui.telemetryIterations.textContent = String(state.iterations);
  updateSeedReadout();
}

function updateModeCopy() {
  const isJulia = state.mode === "julia";
  ui.modeBadge.textContent = isJulia ? "Julia Echo Field" : "Mandelbrot Navigator";
  ui.overlayCaption.textContent = isJulia
    ? "Panning explores the current Julia echo. Return to Mandelbrot to steer the seed live."
    : "Hover to sculpt the Julia echo. Click the stage to lock or release the seed.";
  ui.modeHintChip.textContent = isJulia ? "Seed is fixed while you inspect Julia space" : "Cursor feeds Julia scope";
  ui.modeHint.textContent = isJulia
    ? "Switch back to Mandelbrot whenever you want the preview to follow the main cursor again."
    : "The preview follows your pointer while the seed is unlocked.";
}

function updateControls() {
  ui.presetSelect.value = state.activePreset;
  ui.modeSelect.value = state.mode;
  ui.paletteSelect.value = state.palette;
  ui.iterationsRange.value = String(state.iterations);
  ui.iterationsOutput.textContent = String(state.iterations);

  ui.ambientToggle.classList.toggle("is-active", state.ambientMotion);
  ui.ambientToggle.setAttribute("aria-pressed", String(state.ambientMotion));
  ui.ambientToggle.textContent = state.ambientMotion ? "Ambient Motion" : "Ambient Motion Paused";

  const lockLabel = state.juliaLocked ? "Unlock Seed" : "Lock Seed";
  ui.seedLockButton.textContent = lockLabel;
  ui.juliaLockToggle.textContent = lockLabel;
  ui.juliaLockToggle.classList.toggle("is-active", state.juliaLocked);
  ui.juliaLockToggle.setAttribute("aria-pressed", String(state.juliaLocked));

  updateModeCopy();
  updateTelemetry();
}

function populatePresetSelect() {
  const fragment = document.createDocumentFragment();

  Object.keys(presets).forEach((presetName) => {
    const option = document.createElement("option");
    option.value = presetName;
    option.textContent = presetName;
    fragment.append(option);
  });

  ui.presetSelect.append(fragment);
}

function applyPreset(name) {
  const preset = presets[name];
  if (!preset) {
    return;
  }

  state.activePreset = name;
  state.centerX = preset.centerX;
  state.centerY = preset.centerY;
  state.zoom = preset.zoom;
  state.iterations = preset.iterations;
  state.palette = preset.palette;
  state.juliaX = preset.juliaX;
  state.juliaY = preset.juliaY;
  state.juliaLocked = false;

  interaction.hoverX = state.centerX;
  interaction.hoverY = state.centerY;
  updateControls();
}

function clampZoom(value) {
  return Math.min(Math.max(value, 0.6), 500000);
}

function getCameraScale(cameraZoom) {
  return 3.1 / cameraZoom;
}

function complexFromClient(canvas, clientX, clientY, camera) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return { x: camera.centerX, y: camera.centerY };
  }

  const pixelX = ((clientX - rect.left) / rect.width) * canvas.width;
  const pixelY = ((clientY - rect.top) / rect.height) * canvas.height;
  const minDimension = Math.min(canvas.width, canvas.height);
  const scale = getCameraScale(camera.zoom);

  return {
    x: camera.centerX + ((pixelX - canvas.width * 0.5) / minDimension) * scale,
    y: camera.centerY + ((canvas.height * 0.5 - pixelY) / minDimension) * scale,
  };
}

function updateHoverPoint(clientX, clientY) {
  const point = complexFromClient(ui.mainCanvas, clientX, clientY, state);
  interaction.hoverX = point.x;
  interaction.hoverY = point.y;

  if (!state.juliaLocked && state.mode === "mandelbrot") {
    state.juliaX = point.x;
    state.juliaY = point.y;
  }

  updateTelemetry();
  return point;
}

function panByClientDelta(deltaX, deltaY) {
  const rect = ui.mainCanvas.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return;
  }

  const scaleX = ui.mainCanvas.width / rect.width;
  const scaleY = ui.mainCanvas.height / rect.height;
  const minDimension = Math.min(ui.mainCanvas.width, ui.mainCanvas.height);
  const scale = getCameraScale(state.zoom);

  state.centerX -= (deltaX * scaleX * scale) / minDimension;
  state.centerY += (deltaY * scaleY * scale) / minDimension;
}

function setJuliaLock(nextValue) {
  state.juliaLocked = nextValue;

  if (!state.juliaLocked && state.mode === "mandelbrot") {
    state.juliaX = interaction.hoverX;
    state.juliaY = interaction.hoverY;
  }

  updateControls();
}

function handleStageClick(clientX, clientY) {
  if (state.mode !== "mandelbrot") {
    return;
  }

  const point = updateHoverPoint(clientX, clientY);
  state.juliaX = point.x;
  state.juliaY = point.y;
  state.juliaLocked = !state.juliaLocked;
  updateControls();
}

function zoomAroundPoint(clientX, clientY, zoomMultiplier) {
  const before = complexFromClient(ui.mainCanvas, clientX, clientY, state);
  const nextZoom = clampZoom(state.zoom * zoomMultiplier);
  const provisionalCamera = {
    centerX: state.centerX,
    centerY: state.centerY,
    zoom: nextZoom,
  };
  const after = complexFromClient(ui.mainCanvas, clientX, clientY, provisionalCamera);

  state.zoom = nextZoom;
  state.centerX += before.x - after.x;
  state.centerY += before.y - after.y;
  updateHoverPoint(clientX, clientY);
}

function cyclePreset(direction) {
  const names = Object.keys(presets);
  const currentIndex = names.indexOf(state.activePreset);
  const nextIndex = (currentIndex + direction + names.length) % names.length;
  applyPreset(names[nextIndex]);
}

function updateSheetState() {
  const isTouchLayout = window.matchMedia("(pointer: coarse)").matches || window.innerWidth < 920;
  ui.body.classList.toggle("is-touch", isTouchLayout);

  if (isTouchLayout) {
    ui.controlPanel.classList.remove("expanded");
    ui.sheetToggle.setAttribute("aria-expanded", "false");
  } else {
    ui.controlPanel.classList.add("expanded");
    ui.sheetToggle.setAttribute("aria-expanded", "true");
  }
}

function bindControls() {
  ui.sheetToggle.addEventListener("click", () => {
    if (!ui.body.classList.contains("is-touch")) {
      return;
    }

    ui.controlPanel.classList.toggle("expanded");
    ui.sheetToggle.setAttribute(
      "aria-expanded",
      String(ui.controlPanel.classList.contains("expanded")),
    );
  });

  ui.presetSelect.addEventListener("change", (event) => {
    applyPreset(event.target.value);
  });

  ui.modeSelect.addEventListener("change", (event) => {
    state.mode = event.target.value;
    updateControls();
  });

  ui.paletteSelect.addEventListener("change", (event) => {
    state.palette = event.target.value;
    updateControls();
  });

  ui.iterationsRange.addEventListener("input", (event) => {
    state.iterations = Number(event.target.value);
    updateControls();
  });

  ui.ambientToggle.addEventListener("click", () => {
    state.ambientMotion = !state.ambientMotion;
    if (!state.ambientMotion) {
      interaction.frozenTime = interaction.lastFrameSeconds;
    }
    updateControls();
  });

  ui.seedLockButton.addEventListener("click", () => {
    setJuliaLock(!state.juliaLocked);
  });

  ui.juliaLockToggle.addEventListener("click", () => {
    setJuliaLock(!state.juliaLocked);
  });

  ui.resetButton.addEventListener("click", () => {
    applyPreset(state.activePreset);
  });

  ui.mainCanvas.addEventListener("mousemove", (event) => {
    if (!interaction.mouseDrag) {
      updateHoverPoint(event.clientX, event.clientY);
    }
  });

  ui.mainCanvas.addEventListener("mouseleave", () => {
    interaction.hoverX = state.centerX;
    interaction.hoverY = state.centerY;

    if (!state.juliaLocked && state.mode === "mandelbrot") {
      state.juliaX = interaction.hoverX;
      state.juliaY = interaction.hoverY;
    }

    updateTelemetry();
  });

  ui.mainCanvas.addEventListener("mousedown", (event) => {
    if (event.button !== 0) {
      return;
    }

    interaction.mouseDrag = {
      lastX: event.clientX,
      lastY: event.clientY,
      moved: false,
    };
    ui.body.classList.add("is-dragging");
  });

  window.addEventListener("mousemove", (event) => {
    if (!interaction.mouseDrag) {
      return;
    }

    const deltaX = event.clientX - interaction.mouseDrag.lastX;
    const deltaY = event.clientY - interaction.mouseDrag.lastY;

    if (Math.abs(deltaX) + Math.abs(deltaY) > 1) {
      interaction.mouseDrag.moved = true;
    }

    panByClientDelta(deltaX, deltaY);
    interaction.mouseDrag.lastX = event.clientX;
    interaction.mouseDrag.lastY = event.clientY;
    updateHoverPoint(event.clientX, event.clientY);
  });

  window.addEventListener("mouseup", () => {
    if (!interaction.mouseDrag) {
      return;
    }

    interaction.mouseSuppressClick = interaction.mouseDrag.moved;
    interaction.mouseDrag = null;
    ui.body.classList.remove("is-dragging");

    window.setTimeout(() => {
      interaction.mouseSuppressClick = false;
    }, 0);
  });

  ui.mainCanvas.addEventListener("click", (event) => {
    if (interaction.mouseSuppressClick) {
      return;
    }

    if (interaction.clickTimer) {
      window.clearTimeout(interaction.clickTimer);
    }

    interaction.clickTimer = window.setTimeout(() => {
      handleStageClick(event.clientX, event.clientY);
      interaction.clickTimer = null;
    }, 180);
  });

  ui.mainCanvas.addEventListener("dblclick", (event) => {
    if (interaction.clickTimer) {
      window.clearTimeout(interaction.clickTimer);
      interaction.clickTimer = null;
    }

    zoomAroundPoint(event.clientX, event.clientY, 2.4);
  });

  ui.mainCanvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const zoomMultiplier = Math.exp(-event.deltaY * 0.0012);
      zoomAroundPoint(event.clientX, event.clientY, zoomMultiplier);
    },
    { passive: false },
  );

  ui.mainCanvas.addEventListener(
    "touchstart",
    (event) => {
      if (!event.touches.length) {
        return;
      }

      event.preventDefault();

      if (event.touches.length === 1) {
        const touch = event.touches[0];
        interaction.touchGesture = {
          type: "pan",
          lastX: touch.clientX,
          lastY: touch.clientY,
          moved: false,
        };
        updateHoverPoint(touch.clientX, touch.clientY);
      } else if (event.touches.length === 2) {
        const [touchA, touchB] = event.touches;
        const midpointX = (touchA.clientX + touchB.clientX) / 2;
        const midpointY = (touchA.clientY + touchB.clientY) / 2;
        interaction.touchGesture = {
          type: "pinch",
          startDistance: Math.hypot(
            touchA.clientX - touchB.clientX,
            touchA.clientY - touchB.clientY,
          ),
          startMidpointX: midpointX,
          startMidpointY: midpointY,
          startCenterX: state.centerX,
          startCenterY: state.centerY,
          startZoom: state.zoom,
          moved: true,
        };
        updateHoverPoint(midpointX, midpointY);
      }
    },
    { passive: false },
  );

  ui.mainCanvas.addEventListener(
    "touchmove",
    (event) => {
      if (!interaction.touchGesture || !event.touches.length) {
        return;
      }

      event.preventDefault();

      if (interaction.touchGesture.type === "pan" && event.touches.length === 1) {
        const touch = event.touches[0];
        const deltaX = touch.clientX - interaction.touchGesture.lastX;
        const deltaY = touch.clientY - interaction.touchGesture.lastY;

        if (Math.abs(deltaX) + Math.abs(deltaY) > 3) {
          interaction.touchGesture.moved = true;
        }

        panByClientDelta(deltaX, deltaY);
        interaction.touchGesture.lastX = touch.clientX;
        interaction.touchGesture.lastY = touch.clientY;
        updateHoverPoint(touch.clientX, touch.clientY);
        return;
      }

      if (event.touches.length !== 2) {
        return;
      }

      const [touchA, touchB] = event.touches;
      const currentDistance = Math.max(
        Math.hypot(touchA.clientX - touchB.clientX, touchA.clientY - touchB.clientY),
        12,
      );
      const midpointX = (touchA.clientX + touchB.clientX) / 2;
      const midpointY = (touchA.clientY + touchB.clientY) / 2;
      const gesture = interaction.touchGesture;
      const nextZoom = clampZoom(gesture.startZoom * (currentDistance / gesture.startDistance));
      const before = complexFromClient(
        ui.mainCanvas,
        gesture.startMidpointX,
        gesture.startMidpointY,
        {
          centerX: gesture.startCenterX,
          centerY: gesture.startCenterY,
          zoom: gesture.startZoom,
        },
      );
      const after = complexFromClient(ui.mainCanvas, midpointX, midpointY, {
        centerX: gesture.startCenterX,
        centerY: gesture.startCenterY,
        zoom: nextZoom,
      });

      state.zoom = nextZoom;
      state.centerX = gesture.startCenterX + before.x - after.x;
      state.centerY = gesture.startCenterY + before.y - after.y;
      updateHoverPoint(midpointX, midpointY);
    },
    { passive: false },
  );

  ui.mainCanvas.addEventListener(
    "touchend",
    (event) => {
      if (!interaction.touchGesture) {
        return;
      }

      if (interaction.touchGesture.type === "pan" && !interaction.touchGesture.moved) {
        handleStageClick(interaction.touchGesture.lastX, interaction.touchGesture.lastY);
      }

      if (event.touches.length === 1) {
        const touch = event.touches[0];
        interaction.touchGesture = {
          type: "pan",
          lastX: touch.clientX,
          lastY: touch.clientY,
          moved: false,
        };
        updateHoverPoint(touch.clientX, touch.clientY);
        return;
      }

      interaction.touchGesture = null;
    },
    { passive: true },
  );

  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLElement) {
      const tagName = event.target.tagName.toLowerCase();
      if (tagName === "select" || tagName === "input") {
        return;
      }
    }

    if (event.key.toLowerCase() === "r") {
      event.preventDefault();
      applyPreset(state.activePreset);
      return;
    }

    if (event.key.toLowerCase() === "p") {
      event.preventDefault();
      cyclePreset(1);
      return;
    }

    if (event.code === "Space") {
      event.preventDefault();
      state.ambientMotion = !state.ambientMotion;
      if (!state.ambientMotion) {
        interaction.frozenTime = interaction.lastFrameSeconds;
      }
      updateControls();
    }
  });

  window.addEventListener("resize", () => {
    updateSheetState();
  });
}

function showFallback() {
  ui.body.classList.add("webgl-disabled");
  ui.fallback.hidden = false;
}

function renderFrame(timestamp) {
  const seconds = timestamp * 0.001;
  interaction.lastFrameSeconds = seconds;
  const effectiveTime = state.ambientMotion ? seconds : interaction.frozenTime;

  if (mainRenderer) {
    mainRenderer.render({
      mode: state.mode,
      centerX: state.centerX,
      centerY: state.centerY,
      zoom: state.zoom,
      iterations: state.iterations,
      palette: state.palette,
      juliaX: state.juliaX,
      juliaY: state.juliaY,
      time: effectiveTime,
    });
  }

  if (juliaRenderer) {
    juliaRenderer.render({
      mode: "julia",
      centerX: 0.0,
      centerY: 0.0,
      zoom: 1.9,
      iterations: state.iterations,
      palette: state.palette,
      juliaX: state.juliaX,
      juliaY: state.juliaY,
      time: effectiveTime,
    });
  }

  requestAnimationFrame(renderFrame);
}

function init() {
  populatePresetSelect();
  updateSheetState();
  bindControls();
  updateControls();

  try {
    mainRenderer = createRenderer(ui.mainCanvas);
    juliaRenderer = createRenderer(ui.juliaCanvas);
  } catch (error) {
    console.error(error);
  }

  if (!mainRenderer || !juliaRenderer) {
    showFallback();
    return;
  }

  ui.fallback.hidden = true;
  requestAnimationFrame(renderFrame);
}

init();
