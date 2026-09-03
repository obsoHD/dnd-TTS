"""HTTP routers, one module per concern; ``app.main.create_app`` includes each ``router``.

Ownership (docs/M2-contracts.md): ``health``, ``renders``, ``voices`` are the
service builder's; ``say``, ``jobs`` the worker's; ``board`` the board's;
``play``, ``remote`` the player's. Routers reach shared objects through
``request.app.state`` (attribute names documented in ``app/main.py``) rather
than module globals, so a test can build two apps in one process.
"""
