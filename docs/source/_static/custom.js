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
});
