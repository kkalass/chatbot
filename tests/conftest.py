# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Root test configuration.

This module is loaded by pytest before any test module is imported.
Module-level side effects here therefore run before the application code
that executes at import time (e.g. ``configure_tracing`` in ``app.py``).

We disable OTel here so that importing ``src.chatbot.ui.app`` during test
collection does not connect to a live Phoenix / Jaeger instance and pollute
the operator's observability projects with test noise.

Callers who explicitly want tracing during tests (e.g. debugging an
integration test run) can override this by setting ``OTEL_ENABLED=true``
in the shell environment before invoking pytest.  The integration-test
conftest then ensures those spans land in a dedicated ``-integrationtest``
project rather than the live chatbot project.
"""

import os

# Force OTel off for all test runs.  We use a hard assignment rather than
# setdefault because .env may have been sourced into the shell before pytest
# starts, which would leave OTEL_ENABLED=true already in os.environ and make
# setdefault a no-op.  Tests must never ship spans to a live Phoenix instance.
os.environ["OTEL_ENABLED"] = "false"

# Keep test runs independent from the runtime default chat model. The app
# default can change for production quality, while tests should stay stable
# and lightweight on typical local Ollama setups.
os.environ["CHAT_MODEL"] = "llama3.2"
