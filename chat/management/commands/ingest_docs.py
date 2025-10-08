from django.core.management.base import BaseCommand
from chat.models import Document, DocumentChunk
from chat.chunking import chunk_text
from chat.embedding_utils import embed_text

class Command(BaseCommand):
    help = "Chunk documents and generate embeddings"

    def handle(self, *args, **kwargs):
        docs = Document.objects.all()
        for doc in docs:
            self.stdout.write(f"Processing {doc.filename}")
            chunks = chunk_text(doc.raw_text)
            for i, ch in enumerate(chunks):
                emb = embed_text(ch)
                DocumentChunk.objects.create(
                    document=doc, chunk_index=i, text=ch, embedding=emb
                )
        self.stdout.write(self.style.SUCCESS("Embedding generation complete."))

