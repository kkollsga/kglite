//! `AuthValidator` implementations.
//!
//! Phase C.6 ships [`BasicAuthValidator`] for the `--auth basic` CLI
//! mode. `--auth none` doesn't wire any validator — boltr handles
//! LOGON SUCCESS itself when no validator is supplied, accepting any
//! credentials the driver sends.

use async_trait::async_trait;
use boltr::error::BoltError;
use boltr::server::{AuthCredentials, AuthInfo, AuthValidator};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// HTTP-Basic-style validator: a single `(user, pass)` pair from the
/// CLI. Rejects any other credentials with
/// `Neo.ClientError.Security.Unauthorized`.
pub struct BasicAuthValidator {
    user: String,
    user_digest: [u8; 32],
    pass_digest: [u8; 32],
}

impl BasicAuthValidator {
    pub fn new(user: String, pass: String) -> Self {
        let user_digest = credential_digest(Some(&user));
        let pass_digest = credential_digest(Some(&pass));
        Self {
            user,
            user_digest,
            pass_digest,
        }
    }
}

/// Hash the presence marker and credential bytes to a fixed-size value. The
/// marker keeps a missing field distinct from an explicitly empty credential.
fn credential_digest(value: Option<&str>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    match value {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.as_bytes());
        }
        None => hasher.update([0]),
    }
    hasher.finalize().into()
}

/// Compare fixed-size digests without data-dependent early exit.
fn constant_time_digest_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    bool::from(left.ct_eq(right))
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
        // Hash both fields and compare both fixed-size digests before
        // branching, so mismatches do not reveal a matching prefix or which
        // of username/password failed.
        let supplied_user = credential_digest(credentials.principal.as_deref());
        let supplied_pass = credential_digest(credentials.credentials.as_deref());
        let principal_ok = constant_time_digest_eq(&supplied_user, &self.user_digest);
        let credentials_ok = constant_time_digest_eq(&supplied_pass, &self.pass_digest);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn credentials(user: Option<&str>, pass: Option<&str>) -> AuthCredentials {
        AuthCredentials {
            scheme: "basic".into(),
            principal: user.map(str::to_string),
            credentials: pass.map(str::to_string),
        }
    }

    #[tokio::test]
    async fn accepts_only_the_exact_configured_pair() {
        let validator = BasicAuthValidator::new("alice".into(), "secret".into());
        assert!(validator
            .validate(&credentials(Some("alice"), Some("secret")))
            .await
            .is_ok());

        for candidate in [
            credentials(Some("alicf"), Some("secret")),
            credentials(Some("alice"), Some("secreu")),
            credentials(None, Some("secret")),
            credentials(Some("alice"), None),
        ] {
            assert!(validator.validate(&candidate).await.is_err());
        }
    }

    #[tokio::test]
    async fn missing_credentials_do_not_match_configured_empty_strings() {
        let validator = BasicAuthValidator::new(String::new(), String::new());
        assert!(validator
            .validate(&credentials(Some(""), Some("")))
            .await
            .is_ok());
        assert!(validator.validate(&credentials(None, None)).await.is_err());
    }
}
