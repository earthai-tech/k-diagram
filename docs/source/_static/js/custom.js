document.addEventListener("DOMContentLoaded", function () {
  // Select all elements that can be an expiring badge
  const badges = document.querySelectorAll(".new-badge, .new-badge-card");
  const today = new Date();
  const oneWeekInMilliseconds = 7 * 24 * 60 * 60 * 1000;

  badges.forEach(function (badge) {
    const releaseDateStr = badge.getAttribute("data-release-date");
    if (releaseDateStr) {
      const releaseDate = new Date(releaseDateStr);
      const timeDifference = today.getTime() - releaseDate.getTime();

      // If the release was within the last week, show the badge
      if (timeDifference >= 0 && timeDifference < oneWeekInMilliseconds) {
        badge.style.display = "inline-block";
      }
    }
  });
  
  // --- Card Preview Popup Logic with Debugging ---
  console.log("k-diagram custom script loaded."); // DEBUG: Confirms the JS file is running.

  const previewMap = {
    'card--uncertainty': 'uncertainty.png',
    'card--errors': 'errors.png',
    'card--evaluation': 'evaluation.png',
    'card--importance': 'importance.png',
    'card--relationship': 'relationship.png'
  };

  const cards = document.querySelectorAll(".seealso-card");
  console.log(`Found ${cards.length} cards with the '.seealso-card' class.`); // DEBUG: Confirms cards are being selected.

  cards.forEach(card => {
    let previewImageName = null;
    for (const key in previewMap) {
      if (card.classList.contains(key)) {
        previewImageName = previewMap[key];
        break;
      }
    }

    if (previewImageName) {
      const imagePath = `_static/previews/${previewImageName}`;

      card.addEventListener('mouseenter', function() {
        console.log("Hover detected on:", card.className); // DEBUG: Confirms the hover event is firing.
        
        // Check if a popup already exists to prevent duplicates
        if (this.querySelector('.card-preview-popup')) return;

        const popup = document.createElement('div');
        popup.className = 'card-preview-popup';
        popup.innerHTML = `<img src="${imagePath}" alt="Card preview">`;
        this.appendChild(popup);

        setTimeout(() => {
          popup.style.opacity = '1';
          popup.style.bottom = '105%';
        }, 10);
      });

      card.addEventListener('mouseleave', function() {
        const popup = this.querySelector('.card-preview-popup');
        if (popup) {
          popup.style.opacity = '0';
          popup.style.bottom = '95%';
          setTimeout(() => {
            popup.remove();
          }, 200);
        }
      });
    }
  });
});