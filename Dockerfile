# Offizielles, leichtgewichtiges Python-Image nutzen
FROM python:3.10-slim

# Arbeitsverzeichnis im Container festlegen
WORKDIR /app

# Systempakete installieren, die für den Compiler von scikit-learn gebraucht werden könnten
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Requirements kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK Daten vorab herunterladen, damit es beim Starten des Containers nicht fehlschlägt
RUN python -m nltk.downloader punkt

# App-Dateien kopieren
COPY . .

# Port freigeben
EXPOSE 10000

# Start-Kommando für Uvicorn auf Port 10000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
