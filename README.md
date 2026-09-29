# Gaceta RAG Chatbot

Chatbot con arquitectura **RAG** (Retrieval-Augmented Generation) que responde consultas sobre los documentos de la Gaceta Oficial.

## Estado del proyecto

Repositorio inicial. Próximos pasos:

- [ ] Ingesta del corpus (Markdown y PDF) con chunking
- [ ] Generación de embeddings e índice vectorial
- [ ] Pipeline de retrieval (búsqueda semántica)
- [ ] API e interfaz de chat

## Estructura

```
gaceta-rag-chatbot/
├── data/
│   ├── markdown/   # Corpus en Markdown (versionado en git)
│   └── raw/        # PDFs originales y corpus_colab.zip (ignorados por git)
├── pyproject.toml
└── README.md
```

## Requisitos

- Python 3.11 o superior

## Corpus

- `data/markdown/`: 49 documentos de la Gaceta convertidos a Markdown (~16 MB), versionados en este repositorio.
- `data/raw/`: PDFs originales y `corpus_colab.zip` (~430 MB), disponibles localmente pero fuera del control de versiones.
