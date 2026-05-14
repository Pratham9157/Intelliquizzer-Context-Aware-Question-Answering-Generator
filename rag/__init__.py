"""
RAG Module — Web Retrieval + Local Q&A Agent  (v2)
====================================================
WebRetriever  — DuckDuckGo search + BeautifulSoup scraping (free, no API key)
QAAgent       — Local extractive Q&A using deepset/roberta-base-squad2

No API key required.
"""

from __future__ import annotations

import re
import logging
import threading
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
    _DDG_OK = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _DDG_OK = True
    except ImportError:
        _DDG_OK = False
        logger.warning("ddgs not installed — web RAG disabled")

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/121.0 Safari/537.36'
    )
}
_SKIP = {'youtube.com','reddit.com','twitter.com','x.com',
         'facebook.com','instagram.com','tiktok.com'}

_QA_MODEL = "deepset/roberta-base-squad2"
_qa_pipe  = None
_qa_lock  = threading.Lock()


def _get_qa():
    global _qa_pipe
    if _qa_pipe is not None:
        return _qa_pipe
    with _qa_lock:
        if _qa_pipe is not None:
            return _qa_pipe
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading QA model '{_QA_MODEL}' ...")
            _qa_pipe = hf_pipeline("question-answering", model=_QA_MODEL)
            logger.info("QA model ready.")
        except Exception as e:
            logger.error(f"QA model load failed: {e}")
            _qa_pipe = None
    return _qa_pipe


class WebRetriever:
    def search(self, query: str, max_results: int = 5) -> List[dict]:
        if not _DDG_OK:
            return []
        try:
            with DDGS() as ddgs:
                return [
                    {'title': r.get('title',''), 'url': r.get('href',''),
                     'snippet': r.get('body','')}
                    for r in ddgs.text(query, max_results=max_results)
                ]
        except Exception as e:
            logger.warning(f"DDG search failed: {e}")
            return []

    def fetch_page(self, url: str, max_chars: int = 3000) -> str:
        try:
            domain = url.split('/')[2] if '//' in url else ''
            if any(s in domain for s in _SKIP):
                return ''
            resp = requests.get(url, headers=_HEADERS, timeout=6)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'lxml')
            for tag in soup(['script','style','nav','footer','header','aside','form']):
                tag.decompose()
            text = ' '.join(p.get_text(' ', strip=True)
                           for p in soup.find_all('p') if len(p.get_text()) > 60)
            return re.sub(r'\s+', ' ', text[:max_chars]).strip()
        except Exception:
            return ''

    def retrieve(self, query: str, max_results: int = 3) -> str:
        results = self.search(query, max_results=max_results)
        parts: List[str] = []
        for r in results:
            if r.get('snippet'):
                parts.append(f"[{r['title']}]: {r['snippet']}")
            if r.get('url'):
                page = self.fetch_page(r['url'])
                if page:
                    parts.append(page)
        return '\n\n'.join(parts)[:5000]


class QAAgent:
    def __init__(self) -> None:
        self._retriever = WebRetriever()
        _get_qa()
        logger.info("QAAgent initialised (local RoBERTa)")

    def answer(self, question: str, document_text: str = '',
               use_web: bool = True) -> Dict:
        web_ctx = ''
        sources: List[str] = []

        if use_web:
            results = self._retriever.search(question, max_results=4)
            sources = [r['url'] for r in results if r.get('url')]
            web_ctx = self._retriever.retrieve(question, max_results=3)

        parts: List[str] = []
        if document_text.strip():
            parts.append(document_text[:3000])
        if web_ctx:
            parts.append(web_ctx)
        context = ' '.join(parts).strip()

        if not context:
            return {'answer': 'No context available.', 'sources': sources,
                    'used_web': bool(web_ctx), 'score': 0.0}

        pipe = _get_qa()
        if pipe is None:
            return {'answer': _keyword_fallback(question, context),
                    'sources': sources, 'used_web': bool(web_ctx), 'score': 0.0}

        try:
            result = pipe(question=question, context=context[:4096])
            answer_text = result['answer'].strip()
            score = round(result['score'], 4)
            if score < 0.05:
                answer_text += (f"\n\n*(Low confidence — {score:.0%}. "
                                f"The context may not contain a direct answer.)*")
        except Exception as e:
            logger.error(f"QA error: {e}")
            answer_text = _keyword_fallback(question, context)
            score = 0.0

        return {'answer': answer_text, 'sources': sources,
                'used_web': bool(web_ctx), 'score': score}


def _keyword_fallback(question: str, context: str) -> str:
    q_words  = set(re.findall(r'\w+', question.lower()))
    best, bs = context[:200], 0
    for sent in re.split(r'(?<=[.!?])\s+', context):
        if len(sent) < 20:
            continue
        overlap = len(q_words & set(re.findall(r'\w+', sent.lower()))) / max(len(q_words), 1)
        if overlap > bs:
            bs, best = overlap, sent
    return best if bs > 0.1 else "Could not find a relevant answer in the available context."
