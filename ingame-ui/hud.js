(() => {
  'use strict';

  const overlay = document.getElementById('hud');
  const mapName = document.getElementById('map');
  const wholeTier = document.getElementById('whole');
  const sectionTier = document.getElementById('segment');
  const statusText = document.getElementById('state');

  function renderStatus(status) {
    const isFresh = status && Number.isFinite(status.updatedAt)
      && Math.abs(Date.now() - status.updatedAt) < 2000;

    // 본체가 꺼졌거나 응답이 끊기면 지난 정보도 숨긴다.
    if (!isFresh || !status.active) {
      overlay.style.display = 'none';
      return;
    }

    mapName.textContent = status.map || '—';
    wholeTier.textContent = status.whole || '계산 중';
    sectionTier.textContent = status.segment || '계산 중';
    statusText.textContent = status.running ? 'ON' : 'OFF';
    statusText.classList.toggle('running', !!status.running);
    overlay.style.display = 'block';
  }

  async function refreshStatus() {
    try {
      const response = await fetch('http://127.0.0.1:24051/status.json', {
        cache: 'no-store',
        signal: AbortSignal.timeout(1500),
      });
      if (!response.ok) throw new Error('status unavailable');
      renderStatus(await response.json());
    } catch (_) {
      renderStatus(null);
    }
    setTimeout(refreshStatus, 200);
  }

  refreshStatus();
})();
