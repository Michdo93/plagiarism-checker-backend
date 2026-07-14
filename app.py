import os
import time
import io
import re
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from pypdf import PdfReader

app = FastAPI(title="Plagiatsprüfung-API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PlagiarismChecker:
    def __init__(self):
        # max_clients verringern, um Rate-Limits auf Servern vorzubeugen
        self.ddg = DDGS(max_clients=5)

    def is_valid_sentence(self, sentence):
        """Filtert unbrauchbare Fragmente aus (Zahlen, kurze Verzeichniseinträge)."""
        words = sentence.split()
        if len(words) < 10: # Mindestens 10 Wörter für echte Aussagekraft
            return False
        # Wenn der Satz fast nur aus Zahlen/Sonderzeichen besteht (z.B. Inhaltsverzeichnis)
        alpha_words = [w for w in words if re.search('[a-zA-ZäöüÄÖÜß]', w)]
        if len(alpha_words) / len(words) < 0.6:
            return False
        return True

    def split_into_sentences(self, text):
        sentences = nltk.sent_tokenize(text)
        return [s.strip().replace("\n", " ") for s in sentences if self.is_valid_sentence(s)]

    def search_web(self, sentence, max_results=1):
        urls = []
        try:
            # Exakte Suche in Anführungszeichen
            query = f'"{sentence}"'
            # Verwende 'lite' oder 'html' Backend von DDG, das ist stabiler gegen Limits
            results = self.ddg.text(query, backend="lite", max_results=max_results)
            if results:
                for r in results:
                    urls.append(r['href'])
        except Exception as e:
            # Bei Rate Limit nicht abstürzen, sondern einfach leer zurückgeben
            print(f"[Web-Suche] Übersprungen (Rate-Limit oder Timeout bei Satz: '{sentence[:20]}...')")
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
            print(f"[Scholar-Suche] Fehler -> {e}")
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
    filename = file.filename.lower()
    original_text = ""

    try:
        content_bytes = await file.read()
        
        if filename.endswith('.pdf'):
            pdf_stream = io.BytesIO(content_bytes)
            reader = PdfReader(pdf_stream)
            extracted_pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_pages.append(text)
            original_text = "\n".join(extracted_pages)
            
        elif filename.endswith('.txt'):
            original_text = content_bytes.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unterstützt werden nur .pdf und .txt")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fehler beim Einlesen: {str(e)}")

    if not original_text.strip():
        raise HTTPException(status_code=400, detail="Kein lesbarer Text gefunden.")

    checker = PlagiarismChecker()
    sentences = checker.split_into_sentences(original_text)
    
    if not sentences:
        return {"matches": [], "message": "Keine ausreichend langen Sätze zum Scannen gefunden."}

    potential_sources = {}
    
    # Maximal 10 hochgradig qualifizierte Sätze prüfen (spart Anfragen und beugt Limits vor)
    sentences_to_check = sentences[:10]
    
    for idx, sentence in enumerate(sentences_to_check): 
        print(f"Scanne Satz {idx+1}/{len(sentences_to_check)}: {sentence[:40]}...")
        
        # Web-Suche (DuckDuckGo)
        web_urls = checker.search_web(sentence)
        for url in web_urls:
            if url not in potential_sources:
                potential_sources[url] = "web"
        
        # Wissenschafts-Suche (Semantic Scholar - hat exzellente Rate-Limits!)
        sci_papers = checker.search_scientific(sentence)
        for url, snippet in sci_papers:
            if url not in potential_sources:
                potential_sources[url] = ("scholar", snippet)
        
        # Höfliche Pause verdoppelt (1.5 Sekunde), um DDG-Blockaden abzumildern
        time.sleep(1.5)

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

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {"matches": results}
