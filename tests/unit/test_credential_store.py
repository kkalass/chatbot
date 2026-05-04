# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :class:`InMemoryCredentialStore`."""

from src.chatbot.app.credential_store import InMemoryCredentialStore
from src.chatbot.app.protocols import UsernamePasswordCredentials


class TestInMemoryCredentialStore:
    def test_returns_none_when_empty(self) -> None:
        store = InMemoryCredentialStore()
        assert store.get_credentials("vacation_days") is None

    def test_stores_and_retrieves_by_key(self) -> None:
        store = InMemoryCredentialStore()
        store.set_credentials("vacation_days", "alice", "s3cr3t")
        result = store.get_credentials("vacation_days")
        assert result == UsernamePasswordCredentials(username="alice", password="s3cr3t")

    def test_different_keys_are_independent(self) -> None:
        store = InMemoryCredentialStore()
        store.set_credentials("svc_a", "alice", "pw_a")
        store.set_credentials("svc_b", "bob", "pw_b")
        assert store.get_credentials("svc_a") == UsernamePasswordCredentials("alice", "pw_a")
        assert store.get_credentials("svc_b") == UsernamePasswordCredentials("bob", "pw_b")

    def test_set_credentials_replaces_existing(self) -> None:
        store = InMemoryCredentialStore()
        store.set_credentials("k", "old_user", "old_pw")
        store.set_credentials("k", "new_user", "new_pw")
        assert store.get_credentials("k") == UsernamePasswordCredentials("new_user", "new_pw")

    def test_clear_removes_entry(self) -> None:
        store = InMemoryCredentialStore()
        store.set_credentials("k", "u", "p")
        store.clear_credentials("k")
        assert store.get_credentials("k") is None

    def test_clear_unknown_key_is_idempotent(self) -> None:
        store = InMemoryCredentialStore()
        store.clear_credentials("nonexistent")  # must not raise

    def test_clear_only_removes_target_key(self) -> None:
        store = InMemoryCredentialStore()
        store.set_credentials("a", "u1", "p1")
        store.set_credentials("b", "u2", "p2")
        store.clear_credentials("a")
        assert store.get_credentials("a") is None
        assert store.get_credentials("b") is not None
