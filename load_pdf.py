from langchain_community.document_loaders import PyPDFLoader

# PDF ka path batao
pdf_path = "docs/OS_notes.pdf"

# PDF loader banao
loader = PyPDFLoader(pdf_path)

# PDF load karo - yeh har page ko ek separate "document" mein todta hai
pages = loader.load()

# Kitne pages load hue, check karo
print(f"Total pages loaded: {len(pages)}")

# Pehle page ka content dikhao (check karne ke liye ki sahi load hua)
print("\n--- Page 1 ka content (pehle 500 characters) ---")
print(pages[0].page_content[:500])

# Metadata bhi dikhao (jaise page number, source file)
print("\n--- Page 1 ka metadata ---")
print(pages[0].metadata)
