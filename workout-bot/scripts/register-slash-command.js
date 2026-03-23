/**
 * Register global application commands with Discord API v10.
 * Registers /workout and /sub (not guild-specific).
 *
 * Requires in repo-root .env (this file lives in scripts/):
 *   DISCORD_BOT_TOKEN
 *   DISCORD_APPLICATION_ID
 *
 * Usage (from workout-bot directory):
 *   node scripts/register-slash-command.js
 */

const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, "..", ".env") });

const DISCORD_BOT_TOKEN = (process.env.DISCORD_BOT_TOKEN || "").trim();
const DISCORD_APPLICATION_ID = (process.env.DISCORD_APPLICATION_ID || "").trim();
const DISCORD_PUBLIC_KEY = (process.env.DISCORD_PUBLIC_KEY || "").trim();

function printEnvReport() {
  const rows = [
    ["DISCORD_BOT_TOKEN", DISCORD_BOT_TOKEN],
    ["DISCORD_APPLICATION_ID", DISCORD_APPLICATION_ID],
    ["DISCORD_PUBLIC_KEY", DISCORD_PUBLIC_KEY],
  ];
  console.log("\n--- .env check (values hidden) ---");
  for (const [name, val] of rows) {
    const ok = Boolean(val && val !== "undefined");
    console.log(`  ${name}: ${ok ? "set" : "MISSING or empty"}`);
  }
  console.log("");
}

function printInviteUrl() {
  if (!DISCORD_APPLICATION_ID) return;
  const base = "https://discord.com/api/oauth2/authorize";
  const params = new URLSearchParams({
    client_id: DISCORD_APPLICATION_ID,
    scope: "applications.commands bot",
    permissions: "2048",
  });
  console.log("--- OAuth2 invite (applications.commands + bot, Send Messages) ---");
  console.log(`${base}?${params.toString()}`);
  console.log("");
}

const commands = [
  {
    type: 1,
    name: "workout",
    description:
      "Generate today's workout with Gemini and post it to the configured webhook channel.",
  },
  {
    type: 1,
    name: "sub",
    description: "Info about how workout updates are delivered via the webhook channel.",
  },
];

async function main() {
  printEnvReport();

  if (!DISCORD_BOT_TOKEN || !DISCORD_APPLICATION_ID) {
    console.error(
      "Abort: DISCORD_BOT_TOKEN and DISCORD_APPLICATION_ID must be non-empty in .env",
    );
    console.error(`Looked for .env at: ${path.resolve(__dirname, "..", ".env")}`);
    process.exit(1);
  }

  const url = `https://discord.com/api/v10/applications/${DISCORD_APPLICATION_ID}/commands`;

  let res;
  try {
    res = await fetch(url, {
      method: "PUT",
      headers: {
        Authorization: `Bot ${DISCORD_BOT_TOKEN}`,
        "Content-Type": "application/json",
        "User-Agent": "DiscordBot (workout-bot register-slash-command.js)",
      },
      body: JSON.stringify(commands),
    });
  } catch (err) {
    console.error("Network error calling Discord API:", err);
    process.exit(1);
  }

  const rawText = await res.text();
  let body;
  try {
    body = JSON.parse(rawText);
  } catch {
    body = rawText;
  }

  if (!res.ok) {
    console.error("Discord command registration FAILED.");
    console.error(`HTTP ${res.status} ${res.statusText}`);
    console.error(
      typeof body === "string" ? body : JSON.stringify(body, null, 2),
    );
    printInviteUrl();
    process.exit(1);
  }

  console.log("SUCCESS: global slash commands registered.");
  console.log(JSON.stringify(body, null, 2));
  printInviteUrl();
}

main().catch((e) => {
  console.error("Unexpected error:", e);
  process.exit(1);
});
