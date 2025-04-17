from llama_agent import LlamaAgent

# Initialize the Llama agent
agent = LlamaAgent()

# Test query understanding
test_queries = [
    "Find images of cats",
    "Delete screenshots from January 2025",
    "Show me pictures from last week"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    intent = agent.understand_query(query)
    print(f"Intent analysis: {intent}")
    rewritten = agent.rewrite_query(query)
    print(f"Rewritten query: {rewritten}")
