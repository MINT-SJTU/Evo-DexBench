#!/usr/bin/env bash
# Provide a LAN TLS certificate for the Quest 3 Vuer entry point.
#
# WebXR only runs in a secure context, so Quest Browser needs HTTPS/WSS. The
# certificate's subjectAltName must contain the exact IP the headset dials, and
# this machine's LAN IP comes from DHCP without a reservation, so it changes.
# This script resolves the current LAN IP, reuses a valid certificate for it, and
# generates one when the IP is new. It never rewrites a certificate that is still
# valid for the same IP, so the Quest keeps its stored trust decision.
#
# Usage:
#   tools/quest3/create_lan_certificate.sh                # summary
#   tools/quest3/create_lan_certificate.sh --export       # for eval
#   tools/quest3/create_lan_certificate.sh --ip 192.168.10.93
set -euo pipefail

CERT_DIR="${QUEST3_CERT_DIR:-$HOME/.config/evo-dexbench/quest3-tls}"
VALID_DAYS=365
# Refuse a certificate that is about to expire mid-session.
MIN_REMAINING_SECONDS=$((7 * 24 * 3600))

export_mode=0
requested_ip=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --export) export_mode=1; shift ;;
    --ip) requested_ip="${2:-}"; shift 2 ;;
    --help|-h) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "quest3_lan_cert: unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v openssl >/dev/null || { echo "quest3_lan_cert: openssl is required." >&2; exit 1; }

detect_lan_ip() {
  # The default route records the source address the kernel uses for off-link
  # traffic, which is the address the Quest reaches over the same LAN.
  ip -4 route show default 2>/dev/null \
    | sed -n 's/.*[[:space:]]src[[:space:]]\([0-9.]*\).*/\1/p' \
    | head -1
}

lan_ip="${requested_ip:-$(detect_lan_ip)}"
if [[ -z "$lan_ip" ]]; then
  echo "quest3_lan_cert: could not detect a LAN IPv4 address from the default route." >&2
  echo "quest3_lan_cert: pass one explicitly, e.g. --ip 192.168.10.93" >&2
  exit 1
fi
# A VPN or proxy tunnel can own the default route; its address is unreachable
# from the headset, so reject anything outside the private LAN ranges instead of
# silently producing a certificate the Quest can never validate.
if ! [[ "$lan_ip" =~ ^192\.168\.[0-9]+\.[0-9]+$ || "$lan_ip" =~ ^10\.[0-9]+\.[0-9]+\.[0-9]+$ \
      || "$lan_ip" =~ ^172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+$ ]]; then
  echo "quest3_lan_cert: '$lan_ip' is not a private LAN address (a VPN or proxy" >&2
  echo "quest3_lan_cert: tunnel may own the default route). Pass --ip explicitly." >&2
  exit 1
fi

cert_path="$CERT_DIR/lan-$lan_ip.pem"
key_path="$CERT_DIR/lan-$lan_ip-key.pem"
mkdir -p "$CERT_DIR"
chmod 700 "$CERT_DIR"

certificate_is_usable() {
  [[ -s "$cert_path" && -s "$key_path" ]] || return 1
  openssl x509 -in "$cert_path" -noout -checkend "$MIN_REMAINING_SECONDS" >/dev/null 2>&1 || return 1
  openssl x509 -in "$cert_path" -noout -ext subjectAltName 2>/dev/null \
    | grep -qF "IP Address:$lan_ip"
}

status="reused"
if ! certificate_is_usable; then
  status="generated"
  openssl req -x509 -newkey rsa:2048 -sha256 -nodes \
    -days "$VALID_DAYS" \
    -keyout "$key_path" \
    -out "$cert_path" \
    -subj "/CN=$lan_ip" \
    -addext "subjectAltName=IP:$lan_ip" \
    -addext "basicConstraints=critical,CA:FALSE" \
    -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
    -addext "extendedKeyUsage=serverAuth" \
    >/dev/null 2>&1
  chmod 600 "$key_path"
  chmod 644 "$cert_path"
fi

expires_on="$(openssl x509 -in "$cert_path" -noout -enddate | cut -d= -f2)"

if [[ "$export_mode" == "1" ]]; then
  # Emit commands that survive into child processes after eval. Plain shell
  # assignments would remain unexported and would not reach the Python collector.
  printf 'export QUEST3_LAN_IP=%q\n' "$lan_ip"
  printf 'export QUEST3_CERT=%q\n' "$cert_path"
  printf 'export QUEST3_KEY=%q\n' "$key_path"
  printf 'export QUEST3_CERT_STATUS=%q\n' "$status"
  exit 0
fi

echo "[quest3-cert] LAN IP:      $lan_ip ($status)"
echo "[quest3-cert] certificate: $cert_path"
echo "[quest3-cert] private key: $key_path"
echo "[quest3-cert] expires:     $expires_on"
echo
if [[ "$status" == "generated" ]]; then
  echo "[quest3-cert] This certificate is new, so Quest Browser must accept it once:"
  echo "[quest3-cert]   1. open https://$lan_ip:8013 and accept the warning"
  echo "[quest3-cert]   2. open https://vuer.ai?ws=wss://$lan_ip:8013"
else
  echo "[quest3-cert] Quest entry point: https://vuer.ai?ws=wss://$lan_ip:8013"
fi
