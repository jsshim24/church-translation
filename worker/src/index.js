const WAITING_HTML = `<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Translation</title>
<style>
  html,body{margin:0;height:100%;background:#000;color:#eee;font-family:system-ui,sans-serif}
  #wrap{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px}
  .spinner{width:32px;height:32px;border:3px solid #333;border-top-color:#fff;border-radius:50%;animation:spin 1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .msg{font-size:18px;opacity:.85}
  .sub{font-size:14px;opacity:.5}
</style></head><body>
<div id="wrap">
  <div class="spinner"></div>
  <div class="msg">Waiting for transcription…</div>
  <div class="sub">This page will refresh automatically.</div>
</div>
<script>
(function(){
  async function check(){
    try{
      const r = await fetch('/api/latest', {cache:'no-store'});
      if (r.ok) { location.reload(); return; }
    } catch(e) {}
    setTimeout(check, 1000);
  }
  check();
})();
</script></body></html>`;

const waitingResponse = () => new Response(WAITING_HTML, {
  status: 200,
  headers: {
    'Content-Type': 'text/html; charset=utf-8',
    'Cache-Control': 'no-cache',
  },
});

export default {
  async fetch(request) {
    // Route is `/*` so URLs with query strings (e.g. `/?mode=translation&lang=en`)
    // also hit the Worker — Cloudflare route patterns can't match "root with any
    // query string" directly. Non-root paths pass through unchanged; `/api/*` is
    // additionally exempted from Workers via a Dashboard-configured route (see
    // worker/README.md) so polling traffic doesn't consume free-tier invocations.
    if (new URL(request.url).pathname !== '/') return fetch(request);
    try {
      const resp = await fetch(request);
      if (resp.status >= 500) return waitingResponse();
      return resp;
    } catch (e) {
      return waitingResponse();
    }
  },
};
