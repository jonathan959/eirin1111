/**
 * Cloudflare Worker — Discord slash `/workout` (Interaction endpoint).
 *
 * Deploy:
 *   npm install
 *   npx wrangler login
 *   npx wrangler secret put GEMINI_API_KEY
 *   npx wrangler secret put DISCORD_WEBHOOK_URL
 *   npx wrangler secret put DISCORD_PUBLIC_KEY
 *   npx wrangler secret put DISCORD_APPLICATION_ID
 *   npx wrangler secret put HISTORY_JSON_URL
 *   npx wrangler deploy
 *
 * Discord Developer Portal → Application → Interactions Endpoint URL:
 *   https://<your-worker>.<subdomain>.workers.dev/
 *
 * Register `/workout` once (from your machine, not the Worker):
 *   node scripts/register-slash-command.js
 *
 * HISTORY_JSON_URL: raw GitHub URL to `data/history.json` in your repo
 *   e.g. https://raw.githubusercontent.com/OWNER/REPO/main/data/history.json
 *
 * Note: This path does not push history back to GitHub; the daily GitHub Action
 * is the source of truth for advancing `history.json`. The Worker reads the
 * latest file from HISTORY_JSON_URL for context only.
 */

import { verifyKey } from "discord-interactions";
import { Router } from "itty-router";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const profile = require("../config/profile.js");
const { computeRunContext } = require("../src/run-context.js");
const { generateWorkoutFromContext } = require("../src/workout-gen.js");
const { buildWebhookPayload } = require("../src/discord.js");

const router = Router();

async function fetchHistory(url) {
  if (!url) throw new Error("HISTORY_JSON_URL is not configured");
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`History fetch failed: ${res.status}`);
  return res.json();
}

async function patchInteractionResponse(env, interaction, body) {
  const appId = env.DISCORD_APPLICATION_ID;
  const token = interaction.token;
  const url = `https://discord.com/api/v10/webhooks/${appId}/${token}/messages/@original`;
  const res = await fetch(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Discord follow-up failed: ${res.status} ${t}`);
  }
}

async function postWebhook(env, payload) {
  const url = env.DISCORD_WEBHOOK_URL;
  if (!url) throw new Error("DISCORD_WEBHOOK_URL is not set");
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Webhook post failed: ${res.status} ${t}`);
  }
}

async function handleSub(env, interaction, ctx) {
  try {
    await patchInteractionResponse(env, interaction, {
      content:
        "Workout posts go to the **webhook** channel configured for this bot. Use **`/workout`** anytime to generate today's session. (GitHub Actions can also post on a schedule.)",
    });
  } catch (e) {
    console.error(e);
    try {
      await patchInteractionResponse(env, interaction, {
        content: `❌ Error: ${e.message || String(e)}`,
      });
    } catch (_) {
      /* ignore */
    }
  }
}

async function handleWorkout(env, interaction, ctx) {
  try {
    const history = await fetchHistory(env.HISTORY_JSON_URL);
    const runCtx = computeRunContext(history, profile);
    const workout = await generateWorkoutFromContext(runCtx, env.GEMINI_API_KEY);
    workout.week = runCtx.programWeek;
    workout.phase = runCtx.phase;
    workout.mesocycle = runCtx.mesocycle;

    await postWebhook(env, buildWebhookPayload(workout));

    await patchInteractionResponse(env, interaction, {
      content: "✅ Workout generated and posted to the webhook channel.",
    });
  } catch (e) {
    console.error(e);
    try {
      await patchInteractionResponse(env, interaction, {
        content: `❌ Error: ${e.message || String(e)}`,
      });
    } catch (_) {
      /* ignore */
    }
  }
}

router.post("/", async (request, env, ctx) => {
  const body = await request.text();
  const sig = request.headers.get("X-Signature-Ed25519");
  const ts = request.headers.get("X-Signature-Timestamp");

  const ok = await verifyKey(body, sig, ts, env.DISCORD_PUBLIC_KEY);
  if (!ok) {
    return new Response("invalid request signature", { status: 401 });
  }

  let interaction;
  try {
    interaction = JSON.parse(body);
  } catch {
    return new Response("bad json", { status: 400 });
  }

  if (interaction.type === 1) {
    return new Response(JSON.stringify({ type: 1 }), {
      headers: { "Content-Type": "application/json" },
    });
  }

  if (interaction.type === 2 && interaction.data?.name === "workout") {
    ctx.waitUntil(handleWorkout(env, interaction, ctx));
    return new Response(JSON.stringify({ type: 5 }), {
      headers: { "Content-Type": "application/json" },
    });
  }

  if (interaction.type === 2 && interaction.data?.name === "sub") {
    ctx.waitUntil(handleSub(env, interaction, ctx));
    return new Response(JSON.stringify({ type: 5 }), {
      headers: { "Content-Type": "application/json" },
    });
  }

  return new Response(JSON.stringify({ type: 4, data: { content: "Unknown command" } }), {
    headers: { "Content-Type": "application/json" },
  });
});

router.all("*", () => new Response("Workout bot worker — POST / for Discord interactions", { status: 404 }));

export default {
  fetch: (request, env, ctx) => router.handle(request, env, ctx),
};
