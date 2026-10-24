import uuid


def init(rails):
    # Public rail hook: once per runtime construction, no private cache access.
    print(f"POC_RUNTIME_CREATED {uuid.uuid4().hex}", flush=True)
