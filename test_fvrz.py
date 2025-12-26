from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input="Gib mir Tabellenplatz, Tore (für:gegen) und Spiele für FC Wetzikon 1. Mannschaft.",
    tools=[
        {
            "type": "web_search"
        }
    ]
)

print(response.output_text)
