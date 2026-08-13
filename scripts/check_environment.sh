#!/usr/bin/env bash
set -euo pipefail

export PATH="${PATH:-/usr/bin:/bin}"
case ":$PATH:" in
  *:/usr/bin:*) ;;
  *) PATH="/usr/bin:$PATH" ;;
esac
case ":$PATH:" in
  *:/bin:*) ;;
  *) PATH="/bin:$PATH" ;;
esac

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_PATH" >&2
  exit 1
fi

output_path="$1"

json_escape() {
  local value="${1:-}"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "$value"
}

command_path() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
  fi
}

command_version() {
  local name="$1"
  shift || true
  if command -v "$name" >/dev/null 2>&1; then
    "$name" "$@" 2>/dev/null | head -n 1 | tr -d '\r'
  fi
}

python_json() {
  local found path version launcher
  found=false
  path=""
  version=""
  launcher=""

  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      found=true
      launcher="$candidate"
      path="$(command -v "$candidate")"
      version="$("$candidate" --version 2>/dev/null | head -n 1 | tr -d '\r')"
      break
    fi
  done

  printf '{'
  printf '"found":%s,' "$found"
  printf '"launcher":"%s",' "$(json_escape "$launcher")"
  printf '"path":"%s",' "$(json_escape "$path")"
  printf '"version":"%s"' "$(json_escape "$version")"
  printf '}'
}

git_json() {
  local found=false path="" version=""
  if command -v git >/dev/null 2>&1; then
    found=true
    path="$(command -v git)"
    version="$(git --version 2>/dev/null | head -n 1 | tr -d '\r')"
  fi

  printf '{'
  printf '"found":%s,' "$found"
  printf '"path":"%s",' "$(json_escape "$path")"
  printf '"version":"%s"' "$(json_escape "$version")"
  printf '}'
}

gpu_json() {
  local source="none"
  local detected=false
  local gpu_entries=""

  if command -v nvidia-smi >/dev/null 2>&1; then
    source="nvidia-smi"
    while IFS= read -r row; do
      [[ -z "$row" ]] && continue
      detected=true
      IFS=',' read -r name driver memory <<<"$row"
      name="$(printf '%s' "$name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      driver="$(printf '%s' "$driver" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      memory="$(printf '%s' "$memory" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      local entry
      entry=$(printf '{"name":"%s","driver_version":"%s","memory_mb":%s}' \
        "$(json_escape "$name")" \
        "$(json_escape "$driver")" \
        "$(json_escape "$memory")")
      if [[ -n "$gpu_entries" ]]; then
        gpu_entries+=","
      fi
      gpu_entries+="$entry"
    done < <(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
  elif compgen -G "/dev/nvidia*" >/dev/null 2>&1; then
    source="device-nodes"
    detected=true
  fi

  printf '{'
  printf '"detected":%s,' "$detected"
  printf '"source":"%s",' "$(json_escape "$source")"
  printf '"gpus":[%s]' "$gpu_entries"
  printf '}'
}

wsl_json() {
  local is_wsl=false
  local kernel
  kernel="$(uname -r 2>/dev/null || true)"
  if grep -qi "microsoft" /proc/version 2>/dev/null || [[ "$kernel" == *microsoft* ]] || [[ "$kernel" == *WSL* ]]; then
    is_wsl=true
  fi

  printf '{'
  printf '"visible":%s,' "$is_wsl"
  printf '"kernel":"%s",' "$(json_escape "$kernel")"
  printf '"wsl_exe_on_path":"%s"' "$(json_escape "$(command_path wsl.exe)")"
  printf '}'
}

mkdir -p "$(dirname "$output_path")"

generated_at="$(date -Iseconds)"
os_name="$(uname -s 2>/dev/null || true)"
os_release="$(uname -r 2>/dev/null || true)"
os_machine="$(uname -m 2>/dev/null || true)"
bash_version="${BASH_VERSION:-unknown}"

{
  printf '{'
  printf '"generated_at":"%s",' "$(json_escape "$generated_at")"
  printf '"script":"check_environment.sh",'
  printf '"host":{'
  printf '"os":{"name":"%s","release":"%s","machine":"%s"},' \
    "$(json_escape "$os_name")" \
    "$(json_escape "$os_release")" \
    "$(json_escape "$os_machine")"
  printf '"bash":{"version":"%s"}' "$(json_escape "$bash_version")"
  printf '},'
  printf '"tools":{'
  printf '"python":%s,' "$(python_json)"
  printf '"git":%s' "$(git_json)"
  printf '},'
  printf '"gpu":%s,' "$(gpu_json)"
  printf '"wsl":%s,' "$(wsl_json)"
  printf '"notes":['
  printf '"%s",' "$(json_escape "This report intentionally omits environment variables and credential-bearing settings.")"
  printf '"%s"' "$(json_escape "PATH visibility does not prove solver usability, drivers, or commercial license availability.")"
  printf ']'
  printf '}'
} >"$output_path"
