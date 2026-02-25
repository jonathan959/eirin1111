/**
 * Voice alerts - Text-to-speech for critical trade notifications.
 * Enable via ENABLE_VOICE_ALERTS=1
 */
(function () {
  if (!window.__ENABLE_VOICE_ALERTS) return;
  const speak = (text) => {
    try {
      if ("speechSynthesis" in window && window.speechSynthesis) {
        const u = new SpeechSynthesisUtterance(String(text).slice(0, 200));
        u.rate = 0.95;
        u.pitch = 1;
        u.volume = 1;
        window.speechSynthesis.speak(u);
      }
    } catch (_) {}
  };
  window.__voiceAlert = speak;
  document.addEventListener("voice-alert", (e) => {
    if (e.detail && e.detail.text) speak(e.detail.text);
  });
})();
