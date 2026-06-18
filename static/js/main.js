/* EcoScan — main.js */

// Mobile nav toggle
function toggleMenu() {
  document.querySelector('.nav-links').classList.toggle('open');
}

// Auto-dismiss flash messages
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => el.remove(), 5000);
  });
});
