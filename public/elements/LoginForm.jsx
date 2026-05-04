// SPDX-FileCopyrightText: 2026 Klas Kalaß
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// LoginForm — displayed via AskElementMessage when a tool requires credentials.
// Props (injected globally by Chainlit):
//   props.service_name  — already-translated service name string
//   props.lang          — BCP 47 primary subtag ("de" | "en" | …)

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useState } from "react";

const LABELS = {
  de: {
    title: (s) => `Anmeldung: ${s}`,
    username: "Benutzername",
    password: "Passwort",
    submit: "Anmelden",
    cancel: "Abbrechen",
  },
  en: {
    title: (s) => `Login: ${s}`,
    username: "Username",
    password: "Password",
    submit: "Log in",
    cancel: "Cancel",
  },
};

export default function LoginForm() {
  const lang = (props.lang || "en").slice(0, 2);
  const t = LABELS[lang] || LABELS["en"];

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const canSubmit = username.trim() !== "" && password !== "";

  const handleSubmit = () => {
    if (!canSubmit) return;
    submitElement({ username: username.trim(), password });
  };

  return (
    <Card id="login-form" className="mt-4 w-full max-w-sm">
      <CardHeader>
        <CardTitle>{t.title(props.service_name || "")}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <Label htmlFor="login-username">{t.username}</Label>
          <Input
            id="login-username"
            type="text"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="login-password">{t.password}</Label>
          <Input
            id="login-password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          />
        </div>
      </CardContent>
      <CardFooter className="flex justify-end gap-2">
        <Button
          id="login-cancel"
          variant="outline"
          onClick={() => cancelElement()}
        >
          {t.cancel}
        </Button>
        <Button
          id="login-submit"
          disabled={!canSubmit}
          onClick={handleSubmit}
        >
          {t.submit}
        </Button>
      </CardFooter>
    </Card>
  );
}
