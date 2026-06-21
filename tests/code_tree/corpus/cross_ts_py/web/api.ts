export function createSession(): Promise<Response> {
  return fetch("/api/session", { method: "POST" });
}

export function fetchOther(): Promise<Response> {
  return fetch("/api/nope");
}
