function send(res, status, data) {
  res.status(status).setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  return res.json(data);
}

export default function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return send(res, 405, { detail: "Method not allowed." });
  }

  const configured = Boolean(process.env.GROQ_API_KEY && process.env.KASR1_ACCESS_TOKEN);
  return send(res, 200, {
    service: "kasr-1-hosted-chat-bridge",
    model_loaded: configured,
    provider: "Groq",
    provider_model: "openai/gpt-oss-20b",
    kasr1_weights_loaded: false,
    detail: configured
      ? "Cloud chat is configured. Replies currently use a hosted model; Kasr-1's own trained weights are not connected yet."
      : "Add GROQ_API_KEY and KASR1_ACCESS_TOKEN in Vercel Project Settings → Environment Variables, then redeploy.",
  });
}
