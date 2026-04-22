# Cloudflare Worker: waiting page

Serves a "Waiting for transcription…" page on `https://live.rctranslation.org/` whenever the tunnel has no healthy origin (i.e. `soniox_claude.py` isn't running). Auto-reloads into live captions as soon as the tunnel comes back online.

## Deploy (one-time)

```bash
npm install -g wrangler
wrangler login                # sign in to the same Cloudflare account that owns the tunnel
cd worker
wrangler deploy
```

After deploy, nothing more to do — the Worker lives on Cloudflare's edge 24/7. Any device that runs `soniox_claude.py` (with the `cloudflared` tunnel credentials set up) will be detected as the origin automatically.

## One-time Dashboard step: exempt `/api/*` from Workers

**Do this immediately after `wrangler deploy`, before any viewers load the caption page.**

1. Cloudflare Dashboard → select the `rctranslation.org` zone → Workers Routes → **Add route**.
2. Route: `live.rctranslation.org/api/*`
3. Worker: **None** (leave unassigned / select "disable").
4. Save.

You should now have two routes on the zone:

| Pattern | Worker |
|---------|--------|
| `live.rctranslation.org/*` | `church-waiting-room` |
| `live.rctranslation.org/api/*` | None |

### Why this is needed

The Worker route is `live.rctranslation.org/*` (wildcard) so URLs like `live.rctranslation.org/?mode=translation&lang=en` also show the waiting page when the tunnel is down — Cloudflare route patterns can't express "root with any query string" any other way (`?` isn't allowed in patterns, `*` is the only wildcard).

Without the `/api/*` exemption, every `/api/latest` poll from the in-page caption JS (~6 requests/second per active viewer) would count against the Workers free tier (100,000 requests/day). A couple of viewers over one service window would exhaust it.

Cloudflare resolves routes by specificity — the longest-matching pattern wins. `live.rctranslation.org/api/*` is more specific than `live.rctranslation.org/*`, so `/api/latest` polls match the no-Worker route and go straight to the tunnel. Only root page loads invoke the Worker.

### When this needs to be redone

The Dashboard route lives in Cloudflare's account state, not in this repo. It must be reconfigured if:

- The `rctranslation.org` zone is migrated to a different Cloudflare account.
- The zone is deleted and recreated (e.g., during a registrar change that requires re-onboarding to Cloudflare).
- The polling path in `soniox_claude.py` is renamed away from `/api/...` (e.g., to `/data/latest`). Currently hardcoded in three places: `soniox_claude.py` `poll` function (in-page JS poller), `soniox_claude.py` `_CaptionHandler` class (server handler), and `worker/src/index.js` (waiting-page poller). Adding *new* `/api/...` endpoints doesn't require a Dashboard change — the `/api/*` wildcard already covers them.

For the first two, just repeat the four-step setup above (~30 seconds). For a polling-path rename, update the Dashboard pattern to match the new path alongside the code edits.

## Updating the waiting page

Edit `src/index.js`, then re-run `wrangler deploy`.
