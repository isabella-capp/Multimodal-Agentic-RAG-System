ANSWER_FORMAT = """\
Reply with ONLY the answer, as short as possible — usually a single word or a \
short noun phrase (1-4 words). NEVER write a full sentence, NEVER restate the \
question, NEVER use markdown, NEVER add explanations or phrases such as "Based on \
the context". For example, answer "1889", not "The tower was completed in 1889."; \
answer "granite", not "It is built from granite.". If several answers apply, \
output only the answers separated by commas, like "copper, zinc"."""

# A — no retrieval: the model answers from the image and its own knowledge.
NO_RAG_PROMPT = f"""\
Answer the question about the image.

--- QUESTION ---
{{question}}

{ANSWER_FORMAT}"""

# B — retrieval: the model answers from the image and the retrieved paragraphs.
RAG_PROMPT = f"""\
Answer the question using the image and the context below. Use only information \
that is in the context or visible in the image.

--- CONTEXT ---
{{context}}

--- QUESTION ---
{{question}}

{ANSWER_FORMAT}"""
