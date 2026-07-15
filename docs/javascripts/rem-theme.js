(() => {
  const supportsMotion =
    window.matchMedia("(pointer: fine)").matches &&
    !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (!supportsMotion) {
    return;
  }

  const initialize = () => {
    if (document.querySelector(".rem-cursor-glow")) {
      return;
    }

    const root = document.documentElement;
    const cursorGlow = document.createElement("span");
    cursorGlow.className = "rem-cursor-glow";
    cursorGlow.setAttribute("aria-hidden", "true");
    document.body.appendChild(cursorGlow);

    let latestPointer = null;
    let animationFrame = 0;
    let lastParticleAt = 0;

    const createManaParticle = (x, y) => {
      const existingParticles = document.querySelectorAll(".rem-mana-particle");
      if (existingParticles.length >= 24) {
        existingParticles[0].remove();
      }

      const particle = document.createElement("span");
      const size = 3 + Math.random() * 5;
      particle.className = "rem-mana-particle";
      particle.setAttribute("aria-hidden", "true");
      particle.style.left = x + "px";
      particle.style.top = y + "px";
      particle.style.setProperty("--particle-size", size + "px");
      particle.style.setProperty(
        "--particle-drift-x",
        (Math.random() - 0.5) * 34 + "px",
      );
      particle.style.setProperty(
        "--particle-drift-y",
        -16 - Math.random() * 28 + "px",
      );
      document.body.appendChild(particle);
      particle.addEventListener("animationend", () => particle.remove(), {
        once: true,
      });
    };

    const renderPointer = () => {
      animationFrame = 0;
      if (!latestPointer) {
        return;
      }

      const { clientX, clientY } = latestPointer;
      root.style.setProperty("--rem-pointer-x", clientX + "px");
      root.style.setProperty("--rem-pointer-y", clientY + "px");
      cursorGlow.style.left = clientX + "px";
      cursorGlow.style.top = clientY + "px";
      cursorGlow.style.opacity = "1";

      const hero = document.querySelector(".rem-hero");
      if (hero) {
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
          (0.5 - relativeY) * 3 + "deg",
        );
        hero.style.setProperty(
          "--rem-tilt-y",
          (relativeX - 0.5) * 4 + "deg",
        );
      }

      const now = performance.now();
      if (now - lastParticleAt > 78) {
        createManaParticle(clientX, clientY);
        lastParticleAt = now;
      }
    };

    window.addEventListener(
      "pointermove",
      (event) => {
        latestPointer = event;
        if (!animationFrame) {
          animationFrame = window.requestAnimationFrame(renderPointer);
        }
      },
      { passive: true },
    );

    document.documentElement.addEventListener("mouseleave", () => {
      cursorGlow.style.opacity = "0";
      const hero = document.querySelector(".rem-hero");
      if (hero) {
        hero.style.setProperty("--rem-tilt-x", "0deg");
        hero.style.setProperty("--rem-tilt-y", "0deg");
      }
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
