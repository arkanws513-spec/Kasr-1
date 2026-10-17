import { timingSafeEqual } from "node:crypto";

const MODEL = "openai/gpt-oss-20b";
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

function sameSecret(a, b) {
  const left = Buffer.from(a || "", "utf8");
  const right = Buffer.from(b || "", "utf8");
  return left.length === right.length && timingSafeEqual(left, right);
}

function send(res, status, data) {
  res.status(status).setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  return res.json(data);
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return send(res, 405, { detail: "Method not allowed." });
  }

  const apiKey = process.env.GROQ_API_KEY;
  const accessToken = process.env.KASR1_ACCESS_TOKEN;
  if (!apiKey || !accessToken) {
    return send(res, 503, {
      detail: "Cloud chat is not configured. Add GROQ_API_KEY and KASR1_ACCESS_TOKEN to Vercel Environment Variables.",
    });
  }

  const authorization = req.headers.authorization || "";
  const supplied = authorization.startsWith("Bearer ") ? authorization.slice(7).trim() : "";
  if (!sameSecret(supplied, accessToken)) {
    return send(res, 401, { detail: "Access token is invalid." });
  }

  let body = req.body;
  if (typeof body === "string") {
    try { body = JSON.parse(body); } catch {
      return send(res, 400, { detail: "Request body must be valid JSON." });
    }
  }

  if (!body || !Array.isArray(body.messages) || body.messages.length < 1) {
    return send(res, 400, { detail: "messages must be a non-empty array." });
  }

  const messages = body.messages
    .filter((m) => m && (m.role === "user" || m.role === "assistant") &&
      typeof m.content === "string" && m.content.trim())
    .slice(-12)
    .map((m) => ({ role: m.role, content: m.content.trim().slice(0, 12000) }));

  if (!messages.length) return send(res, 400, { detail: "No valid messages were supplied." });

  try {
    const upstream = await fetch(GROQ_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          {
            role: "system",
            content: "You are the temporary hosted conversation engine connected to the Kasr-1 interface. Answer helpfully and accurately in the user's language, including Arabic or English. Do not claim to be running Kasr-1's own trained weights; those will be integrated separately after training.",
          },
          ...messages,
        ],
        max_tokens: 600,
        temperature: 0.7,
      }),
      signal: AbortSignal.timeout(45000),
    });

    const result = await upstream.json().catch(() => ({}));
    if (!upstream.ok) {
      const detail = result?.error?.message || "The hosted model provider returned an error.";
      return send(res, upstream.status === 429 ? 429 : 502, { detail });
    }

    const response = result?.choices?.[0]?.message?.content;
    if (typeof response !== "string" || !response.trim()) {
      return send(res, 502, { detail: "The hosted model returned an empty response." });
    }

    return send(res, 200, {
      response: response.trim(),
      provider: "Groq",
      model: MODEL,
      kasr1_weights_loaded: false,
    });
  } catch (error) {
    const timedOut = error?.name === "TimeoutError" || error?.name === "AbortError";
    return send(res, 502, {
      detail: timedOut ? "The hosted model request timed out. Try again." : "Could not reach the hosted model provider.",
    });
  }
}
