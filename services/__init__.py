"""Backend services for Eirin Bot.

Each submodule is a thin, testable service that the API layer (one_server.py,
worker_api.py) can call. Keeping them small and side-effect free makes them
easy to unit test and reason about.
"""
