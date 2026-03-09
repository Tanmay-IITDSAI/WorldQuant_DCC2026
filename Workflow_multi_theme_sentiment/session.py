import time
from typing import Any

import requests
from requests.auth import HTTPBasicAuth


class BrainSession(requests.Session):
    def __init__(
            self,
            base_url: str,
            email: str,
            password: str,
            refresh_margin: float = 5.0,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.auth_url = f"{self.base_url}/authentication"
        self.email = email
        self.password = password
        self.refresh_margin = refresh_margin
        self._expiry_ts: float = 0.0

    def _is_expired(self) -> bool:
        if self._expiry_ts <= 0:
            return True
        return time.time() >= (self._expiry_ts - self.refresh_margin)

    def _update_expiry_from_response(self, response: requests.Response) -> None:
        data = response.json()
        token_obj = data.get("token")
        if not isinstance(token_obj, dict):
            raise ValueError("Authentication response missing 'token' object")

        expiry_seconds = token_obj.get("expiry")
        if expiry_seconds is None:
            raise ValueError("Authentication response token missing 'expiry'")

        self._expiry_ts = time.time() + float(expiry_seconds)

    def _persona_2fa_flow(self, location: str) -> None:
        if location.startswith("/"):
            url = f"{self.base_url}{location}"
        else:
            url = location

        print("Biometrics authentication required. Open this URL in your browser and complete 2FA:")
        print(url)

        while True:
            input("Press Enter after you have completed (or retried) 2FA...")

            response = super().post(url)

            if response.status_code == 403:
                print("2FA is not complete or was rejected (received HTTP 403). Please complete 2FA in your browser and try again.")
                continue

            response.raise_for_status()
            self._update_expiry_from_response(response)
            return

    def _authenticate(self) -> None:
        response = super().post(self.auth_url, auth=HTTPBasicAuth(self.email, self.password))

        if response.status_code == 401:
            www_authenticate = response.headers.get("WWW-Authenticate", "")
            if www_authenticate.lower() == "persona":
                location = response.headers.get("Location")
                if not location:
                    raise RuntimeError("Persona 2FA indicated (WWW-Authenticate=persona) but no Location header provided.")
                self._persona_2fa_flow(location)
                return

        response.raise_for_status()
        self._update_expiry_from_response(response)

    def _ensure_authenticated(self) -> None:
        if self._is_expired():
            self._authenticate()

    def request(
            self,
            method: str,
            url: str,
            *args: Any,
            **kwargs: Any,
    ) -> requests.Response:
        # Allow callers to pass just a path like "/users"
        if url.startswith("/"):
            full_url = f"{self.base_url}{url}"
        else:
            full_url = url

        # Do NOT auto-auth for any URL under the authentication path
        if not full_url.startswith(self.auth_url):
            self._ensure_authenticated()

        return super().request(method, full_url, *args, **kwargs)
