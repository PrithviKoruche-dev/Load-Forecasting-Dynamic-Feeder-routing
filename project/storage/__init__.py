"""Database layer (gap-analysis item D1): schema and a repository the other modules write through.

    from storage import Store
    store = Store()                 # URL from config/storage.yaml or LFDR_DB_URL
"""
from storage.store import Store  # noqa: F401
