import os
import time
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

app = FastAPI(title="Plagiatsprüfung-API")

# CORS konfigurieren, damit dein Frontend (auch von localhost oder anderen Domains) darauf zugreifen darf
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NLTK-Tokenizer beim Starten herunterladen
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PlagiarismChecker:
    def __init__(self):
        self.ddg = DDGS()

    def split_into_sentences(self, text, min_words=8):
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.split()) >= min_words]

    def search_web(self, sentence, max_results=2):
        urls = []
        try:
            # Exakte Satzsuche
            query = f'"{sentence}"'
            results = self.ddg.text(query, max_results=max_results)
            if results:
                for r in results:
                    urls.append(r['href'])
        except Exception as e:
            print(f"Fehler bei Web-Suche für Satz: {sentence[:30]} -> {e}")
        return urls

    def search_scientific(self, sentence, max_results=2):
        urls = []
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": sentence, "limit": max_results, "fields": "title,url,abstract"}
        try:
            response = requests.get(base_url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for paper in data.get("data", []):
                    if paper.get("url"):
                        urls.append((paper["url"], paper.get("title") + " - " + (paper.get("abstract") or "")))
        except Exception as e:
            print(f"Fehler bei Scholar-Suche -> {e}")
        return urls

    def scrape_url_text(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text()
        except:
            pass
        return ""

    def calculate_similarity(self, text_a, text_b):
        if not text_a.strip() or not text_b.strip():
            return 0.0
        try:
            vectorizer = TfidfVectorizer().fit_transform([text_a, text_b])
            vectors = vectorizer.toarray()
            similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
            return round(similarity * 100, 2)
        except:
            return 0.0

@app.get("/")
def read_root():
    return {"status": "online", "message": "Plagiatsprüfer-API läuft!"}

@app.post("/scan")
async def scan_document(file: UploadFile = File(...)):
    # Unterstützt momentan einfache .txt-Dateien (für PDFs/Word müsste man extra Parser einbinden)
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Es werden derzeit nur .txt-Dateien unterstützt.")
    
    try:
        content_bytes = await file.read()
        original_text = content_bytes.decode('utf-8')
    except Exception:
        raise HTTPException(status_code=400, detail="Datei konnte nicht als UTF-8 gelesen werden.")

    checker = PlagiarismChecker()
    sentences = checker.split_into_sentences(original_text)
    
    if not sentences:
        return {"matches": [], "message": "Der hochgeladene Text ist zu kurz oder enthält keine brauchbaren Sätze."}

    potential_sources = {}
    
    # Durchsuche die ersten Sätze (begrenzt, um Timeouts auf Render.com zu vermeiden)
    for idx, sentence in enumerate(sentences[:15]): 
        # Web
        web_urls = checker.search_web(sentence)
        for url in web_urls:
            if url not in potential_sources:
                potential_sources[url] = "web"
        
        # Scholar
        sci_papers = checker.search_scientific(sentence)
        for url, snippet in sci_papers:
            if url not in potential_sources:
                potential_sources[url] = ("scholar", snippet)
        
        time.sleep(0.5) # Höfliches Delay

    results = []
    for url, source_type in potential_sources.items():
        source_text = ""
        source_name = url
        
        if source_type == "web":
            source_text = checker.scrape_url_text(url)
        else:
            source_name = f"[Wissenschaft] {source_type[1][:60]}... ({url})"
            source_text = source_type[1]

        if source_text:
            similarity = checker.calculate_similarity(original_text, source_text)
            if similarity > 3.0:
                results.append({"source": source_name, "score": similarity})

    # Sortieren nach Ähnlichkeit
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {"matches": results}
