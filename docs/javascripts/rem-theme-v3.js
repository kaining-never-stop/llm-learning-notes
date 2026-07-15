(() => {
  const STORAGE_KEY = "llm-notes-visual-theme";
  const root = document.documentElement;
  const reducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  let remThemeEnabled = true;
  let particleCanvas = null;
  let particleContext = null;
  let cursorGlow = null;
  let themeSwitch = null;
  let animationFrame = 0;
  let particles = [];
  let lastPointerX = 0;
  let lastPointerY = 0;
  let lastParticleAt = 0;
  let devicePixelRatio = 1;

  const readStoredTheme = () => {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch (_error) {
      return null;
    }
  };

  const storeTheme = (themeName) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, themeName);
    } catch (_error) {
      // The theme still works when local storage is unavailable.
    }
  };

  const updateThemeSwitch = () => {
    if (!themeSwitch) {
      return;
    }

    const label = themeSwitch.querySelector(".rem-theme-switch__label");
    const icon = themeSwitch.querySelector(".rem-theme-switch__icon");
    const targetTheme = remThemeEnabled ? "普通主题" : "雷姆主题";

    label.textContent = targetTheme;
    icon.textContent = remThemeEnabled ? "◐" : "❄";
    themeSwitch.setAttribute(
      "aria-label",
      "切换为" + targetTheme,
    );
    themeSwitch.setAttribute("aria-pressed", String(remThemeEnabled));
    themeSwitch.title = "切换为" + targetTheme;
  };

  const clearParticles = () => {
    particles = [];
    if (particleContext && particleCanvas) {
      particleContext.clearRect(
        0,
        0,
        particleCanvas.clientWidth,
        particleCanvas.clientHeight,
      );
    }
  };

  const setTheme = (enabled, persist = true) => {
    remThemeEnabled = enabled;
    root.dataset.remTheme = enabled ? "on" : "off";

    if (persist) {
      storeTheme(enabled ? "rem" : "plain");
    }

    if (!enabled) {
      clearParticles();
      if (cursorGlow) {
        cursorGlow.style.opacity = "0";
      }
    }

    updateThemeSwitch();
  };

  const resizeParticleCanvas = () => {
    if (!particleCanvas || !particleContext) {
      return;
    }

    devicePixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    particleCanvas.width = Math.floor(window.innerWidth * devicePixelRatio);
    particleCanvas.height = Math.floor(window.innerHeight * devicePixelRatio);
    particleCanvas.style.width = window.innerWidth + "px";
    particleCanvas.style.height = window.innerHeight + "px";
    particleContext.setTransform(
      devicePixelRatio,
      0,
      0,
      devicePixelRatio,
      0,
      0,
    );
  };

  const createParticle = (x, y, travelX, travelY) => {
    const palette = [
      [134, 204, 255],
      [85, 181, 245],
      [223, 241, 255],
      [240, 155, 209],
    ];
    const color = palette[Math.floor(Math.random() * palette.length)];
    const direction = Math.atan2(travelY, travelX);
    const backwards = direction + Math.PI;
    const speed = 0.5 + Math.random() * 1.8;

    particles.push({
      x: x + (Math.random() - 0.5) * 12,
      y: y + (Math.random() - 0.5) * 12,
      velocityX:
        Math.cos(backwards) * speed + (Math.random() - 0.5) * 1.1,
      velocityY:
        Math.sin(backwards) * speed - 0.35 - Math.random() * 0.9,
      size: 2.5 + Math.random() * 5.5,
      rotation: Math.random() * Math.PI,
      rotationSpeed: (Math.random() - 0.5) * 0.14,
      life: 1,
      decay: 0.018 + Math.random() * 0.018,
      color,
    });
  };

  const drawParticle = (particle) => {
    const alpha = Math.max(0, particle.life);
    const [red, green, blue] = particle.color;
    const size = particle.size * (0.45 + alpha * 0.55);

    particleContext.save();
    particleContext.translate(particle.x, particle.y);
    particleContext.rotate(particle.rotation);
    particleContext.globalAlpha = alpha;
    particleContext.fillStyle =
      "rgba(" + red + "," + green + "," + blue + "," + alpha + ")";
    particleContext.shadowBlur = 12;
    particleContext.shadowColor =
      "rgba(" + red + "," + green + "," + blue + ",0.9)";
    particleContext.beginPath();
    particleContext.moveTo(0, -size);
    particleContext.lineTo(size * 0.34, -size * 0.34);
    particleContext.lineTo(size, 0);
    particleContext.lineTo(size * 0.34, size * 0.34);
    particleContext.lineTo(0, size);
    particleContext.lineTo(-size * 0.34, size * 0.34);
    particleContext.lineTo(-size, 0);
    particleContext.lineTo(-size * 0.34, -size * 0.34);
    particleContext.closePath();
    particleContext.fill();
    particleContext.restore();
  };

  const animateParticles = () => {
    animationFrame = 0;
    if (!particleContext || !particleCanvas) {
      return;
    }

    particleContext.clearRect(
      0,
      0,
      particleCanvas.clientWidth,
      particleCanvas.clientHeight,
    );

    particles = particles.filter((particle) => {
      particle.x += particle.velocityX;
      particle.y += particle.velocityY;
      particle.velocityX *= 0.985;
      particle.velocityY = particle.velocityY * 0.985 + 0.012;
      particle.rotation += particle.rotationSpeed;
      particle.life -= particle.decay;

      if (particle.life <= 0) {
        return false;
      }

      drawParticle(particle);
      return true;
    });

    if (particles.length > 0 && remThemeEnabled) {
      animationFrame = window.requestAnimationFrame(animateParticles);
    }
  };

  const startParticleAnimation = () => {
    if (!animationFrame && particles.length > 0) {
      animationFrame = window.requestAnimationFrame(animateParticles);
    }
  };

  const updateHeroParallax = (clientX, clientY) => {
    const hero = document.querySelector(".rem-hero");
    if (!hero) {
      return;
    }

    const bounds = hero.getBoundingClientRect();
    const relativeX = Math.min(
      1,
      Math.max(0, (clientX - bounds.left) / bounds.width),
    );
    const relativeY = Math.min(
      1,
      Math.max(0, (clientY - bounds.top) / bounds.height),
    );
    hero.style.setProperty(
      "--rem-tilt-x",
      (0.5 - relativeY) * 5 + "deg",
    );
    hero.style.setProperty(
      "--rem-tilt-y",
      (relativeX - 0.5) * 7 + "deg",
    );
  };

  const handlePointerMove = (event) => {
    if (!remThemeEnabled || event.pointerType === "touch") {
      return;
    }

    const clientX = event.clientX;
    const clientY = event.clientY;
    const travelX = clientX - lastPointerX;
    const travelY = clientY - lastPointerY;
    const distance = Math.hypot(travelX, travelY);

    root.style.setProperty("--rem-pointer-x", clientX + "px");
    root.style.setProperty("--rem-pointer-y", clientY + "px");

    if (cursorGlow) {
      cursorGlow.style.left = clientX + "px";
      cursorGlow.style.top = clientY + "px";
      cursorGlow.style.opacity = "1";
    }

    updateHeroParallax(clientX, clientY);

    const now = performance.now();
    if (!reducedMotion && distance > 1 && now - lastParticleAt > 12) {
      const particleCount = Math.min(7, Math.max(3, Math.ceil(distance / 9)));
      for (let index = 0; index < particleCount; index += 1) {
        createParticle(clientX, clientY, travelX, travelY);
      }
      lastParticleAt = now;
      startParticleAnimation();
    }

    lastPointerX = clientX;
    lastPointerY = clientY;
  };

  const resetPointerEffects = () => {
    if (cursorGlow) {
      cursorGlow.style.opacity = "0";
    }

    const hero = document.querySelector(".rem-hero");
    if (hero) {
      hero.style.setProperty("--rem-tilt-x", "0deg");
      hero.style.setProperty("--rem-tilt-y", "0deg");
    }
  };

  const updatePageLayout = () => {
    root.classList.toggle(
      "is-landing-page",
      Boolean(document.querySelector(".landing-page-marker")),
    );
  };

  const initialize = () => {
    if (document.querySelector(".rem-theme-switch")) {
      return;
    }

    particleCanvas = document.createElement("canvas");
    particleCanvas.id = "rem-particle-canvas";
    particleCanvas.setAttribute("aria-hidden", "true");
    document.body.appendChild(particleCanvas);
    particleContext = particleCanvas.getContext("2d");

    cursorGlow = document.createElement("span");
    cursorGlow.className = "rem-cursor-glow";
    cursorGlow.setAttribute("aria-hidden", "true");
    document.body.appendChild(cursorGlow);

    themeSwitch = document.createElement("button");
    themeSwitch.className = "rem-theme-switch";
    themeSwitch.type = "button";
    themeSwitch.innerHTML =
      '<span class="rem-theme-switch__icon" aria-hidden="true">◐</span>' +
      '<span class="rem-theme-switch__label">普通主题</span>';
    const headerOption = document.querySelector(".md-header__option");
    if (headerOption && headerOption.parentNode) {
      headerOption.parentNode.insertBefore(themeSwitch, headerOption);
    } else {
      document.body.appendChild(themeSwitch);
    }

    themeSwitch.addEventListener("click", () => {
      setTheme(!remThemeEnabled);
    });

    window.addEventListener("resize", resizeParticleCanvas, { passive: true });
    window.addEventListener("pointermove", handlePointerMove, {
      passive: true,
    });
    document.documentElement.addEventListener("mouseleave", resetPointerEffects);

    resizeParticleCanvas();
    updatePageLayout();
    const storedTheme = readStoredTheme();
    setTheme(storedTheme !== "plain", false);
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(updatePageLayout);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
