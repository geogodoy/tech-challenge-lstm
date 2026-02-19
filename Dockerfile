FROM python:3.10-slim

WORKDIR /app

# Copiar requirements e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código-fonte
COPY src/app.py ./src/
COPY src/model.py ./src/

# Copiar artefatos do modelo
COPY models/ ./models/

# Configurar PYTHONPATH para imports
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando para rodar a API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
