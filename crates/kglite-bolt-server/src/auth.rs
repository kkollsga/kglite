//! `AuthValidator` implementations.
//!
//! Phase C.6 ships [`BasicAuthValidator`] for the `--auth basic` CLI
//! mode. `--auth none` doesn't wire any validator — boltr handles
//! LOGON SUCCESS itself when no validator is supplied, accepting any
//! credentials the driver sends.

use async_trait::async_trait;
use boltr::error::BoltError;
use boltr::server::{AuthCredentials, AuthInfo, AuthValidator};

/// HTTP-Basic-style validator: a single `(user, pass)` pair from the
/// CLI. Rejects any other credentials with
/// `Neo.ClientError.Security.Unauthorized`.
pub struct BasicAuthValidator {
    user: String,
    pass: String,
}

impl BasicAuthValidator {
    pub fn new(user: String, pass: String) -> Self {
        Self { user, pass }
    }
}

#[async_trait]
impl AuthValidator for BasicAuthValidator {
    async fn validate(&self, credentials: &AuthCredentials) -> Result<AuthInfo, BoltError> {
        // Drivers sending `auth=("user", "pass")` use scheme "basic".
        // Reject any other scheme cleanly so users get a clear error
        // rather than a silent "Invalid credentials" mismatch.
        if credentials.scheme != "basic" {
            return Err(BoltError::Authentication(format!(
                "scheme '{}' not supported — kglite-bolt-server --auth basic only accepts 'basic'",
                credentials.scheme
            )));
        }
        let principal_ok = credentials.principal.as_deref() == Some(self.user.as_str());
        let credentials_ok = credentials.credentials.as_deref() == Some(self.pass.as_str());
        if !principal_ok || !credentials_ok {
            return Err(BoltError::Authentication(
                "invalid username or password".into(),
            ));
        }
        Ok(AuthInfo {
            principal: self.user.clone(),
            credentials_expired: false,
        })
    }
}
