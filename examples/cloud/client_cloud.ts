import OpenAI from "openai";

const apiKey = process.env.HANZO_API_KEY;
if (!apiKey) {
  console.error("Error: HANZO_API_KEY environment variable is required");
  console.error("Obtain your key from https://platform.hanzo.ai");
  process.exit(1);
}

const client = new OpenAI({
  apiKey,
  baseURL: "https://api.hanzo.ai/v1",
});

async function main() {
  console.log("Connecting to Hanzo Cloud with TypeScript client...");

  const response = await client.chat.completions.create({
    model: "zen5",
    messages: [
      { role: "system", content: "You are a concise engineering assistant." },
      { role: "user", content: "Explain how Zen routes open-weight models." },
    ],
  });

  console.log("\nResponse from Zen:");
  console.log(response.choices[0].message.content);
}

main().catch(console.error);
