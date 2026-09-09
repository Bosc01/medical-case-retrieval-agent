#!/usr/bin/env bash
# Export macOS trust roots (including any proxy CA) so Python can verify TLS.
# Only needed on machines behind a TLS-intercepting proxy.
set -euo pipefail
out="$(cd "$(dirname "$0")/.." && pwd)/data/ca-roots.pem"
security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain >  "$out"
security find-certificate -a -p /Library/Keychains/System.keychain                        >> "$out"
echo "wrote $(grep -c 'BEGIN CERTIFICATE' "$out") certificates to $out"
