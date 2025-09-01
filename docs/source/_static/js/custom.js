document.addEventListener("DOMContentLoaded", function () {
  /* ============================================================
   * 1) "NEW" badge — show for one week after release date
   * ========================================================== */
  const badges = document.querySelectorAll(".new-badge, .new-badge-card");
  const today = new Date();
  const oneWeekInMilliseconds = 7 * 24 * 60 * 60 * 1000;

  badges.forEach(function (badge) {
    const releaseDateStr = badge.getAttribute("data-release-date");
    if (!releaseDateStr) return;
    const releaseDate = new Date(releaseDateStr);
    const timeDifference = today.getTime() - releaseDate.getTime();
    if (timeDifference >= 0 && timeDifference < oneWeekInMilliseconds) {
      badge.style.display = "inline-block";
    }
  });

  /* ============================================================
   * 2) Card Preview Popups (See-also tiles)
   *    - robust inline styles so it works even without CSS rules
   * ========================================================== */
  console.log("k-diagram custom script loaded.");

  const previewMap = {
    "card--uncertainty":  "uncertainty.png",
    "card--errors":       "errors.png",
    "card--evaluation":   "evaluation.png",
    "card--importance":   "importance.png",
    "card--relationship": "relationship.png"
  };

  const cards = document.querySelectorAll(".seealso-card"); // matches your :class-card
  console.log(`Found ${cards.length} seealso cards.`);

  // Preload images for smooth popups
  const preload = new Set(Object.values(previewMap));
  preload.forEach(name => {
    const img = new Image();
    img.src = `_static/previews/${name}`;
  });

  cards.forEach(card => {
    // Make sure the card can position an absolutely-positioned child
    const computedPos = window.getComputedStyle(card).position;
    if (computedPos === "static") card.style.position = "relative";

    // Determine preview image for this card
    let previewImageName = null;
    for (const cls in previewMap) {
      if (card.classList.contains(cls)) {
        previewImageName = previewMap[cls];
        break;
      }
    }
    if (!previewImageName) return; // Full API card has no preview

    const imagePath = `_static/previews/${previewImageName}`;

    const showPopup = () => {
      if (card.querySelector(".card-preview-popup")) return;

      const popup = document.createElement("div");
      popup.className = "card-preview-popup";
      popup.innerHTML = `<img src="${imagePath}" alt="Card preview">`;

      // Inline styles so it works even if CSS isn’t present
      popup.style.position = "absolute";
      popup.style.left = "50%";
      popup.style.transform = "translateX(-50%)";
      popup.style.bottom = "95%";       // start below the target position
      popup.style.opacity = "0";        // start transparent
      popup.style.transition = "opacity .2s ease, bottom .2s ease";
      popup.style.zIndex = "10";
      popup.style.pointerEvents = "none";
      popup.style.boxShadow = "0 10px 28px rgba(0,0,0,.18)";
      popup.style.borderRadius = "10px";
      popup.style.overflow = "hidden";
      popup.style.background = "transparent";

      const img = popup.querySelector("img");
      img.style.display = "block";
      img.style.maxWidth = "240px";
      img.style.maxHeight = "240px";
      img.style.width = "auto";
      img.style.height = "auto";

      card.appendChild(popup);

      // animate in
      requestAnimationFrame(() => {
        popup.style.opacity = "1";
        popup.style.bottom = "105%";
      });
    };

    const hidePopup = () => {
      const popup = card.querySelector(".card-preview-popup");
      if (!popup) return;
      popup.style.opacity = "0";
      popup.style.bottom = "95%";
      setTimeout(() => popup && popup.remove(), 200);
    };

    // Mouse + keyboard accessibility
    card.addEventListener("mouseenter", showPopup);
    card.addEventListener("mouseleave", hidePopup);
    card.addEventListener("focusin", showPopup);
    card.addEventListener("focusout", hidePopup);
  });

  /* ============================================================
   * 3) Auto-class "Practical examples" admonitions
   *    (also tolerates the "Pratical examples" typo)
   * ========================================================== */
  document.querySelectorAll(".admonition > .admonition-title").forEach(t => {
    const raw = t.textContent.trim();
    const txt = raw.toLowerCase();

    // Normalize whitespace
    const norm = txt.replace(/\s+/g, " ");

    // Match Practical/Pratical Example(s)
    // Accepts: "Practical example(s)", "Pratical example(s)" (any case)
    if (/^pra(c)?tical example(s)?$/.test(norm)) {
      const box = t.parentElement;
      box.classList.add("practical-examples");
      // ribbon text: EXAMPLE vs EXAMPLES
      const plural = /examples$/.test(norm);
      box.setAttribute("data-badge", plural ? "EXAMPLES" : "EXAMPLE");
      return;
    }

    // Match Best practice(s)
    if (/^best practice(s)?$/.test(norm)) {
      const box = t.parentElement;
      box.classList.add("best-practice");
      const plural = /practices$/.test(norm);
      box.setAttribute("data-badge", plural ? "BEST PRACTICES" : "BEST PRACTICE");
      return;
    }
  });

  // Auto-class "Plot Anatomy" (allow suffixes like "(Radar Chart)")
  document.querySelectorAll(".admonition > .admonition-title").forEach(t => {
    const raw  = (t.textContent || t.innerText || "").trim();
    const norm = raw.replace(/\s+/g, " ").toLowerCase();

    // starts with "plot anatomy", then end or any non-alphanumeric punctuation/suffix
    if (/^plot anatomy(?:$|[^a-z0-9_].*)/.test(norm)) {
      const box = t.parentElement;
      box.classList.add("plot-anatomy");
      // if you want a ribbon text
      if (!box.hasAttribute("data-badge")) box.setAttribute("data-badge", "KEY");
    }
  });
});


