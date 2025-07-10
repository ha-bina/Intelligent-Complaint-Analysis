from src.rag_pipeline import rag_pipeline

questions = [
    "Why are people frustrated with BNPL services?",
    "What problems do users face with money transfers?",
    "How do customers describe savings account issues?",
    "Do complaints mention account closures without notice?",
    "Are there concerns about late fees on personal loans?"
]

for q in questions:
    answer, sources = rag_pipeline(q)
    print(f"\nQuestion: {q}\nAnswer: {answer}\nSources: {sources}")
