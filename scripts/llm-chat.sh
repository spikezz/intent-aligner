#!/usr/bin/env bash
# Send single messages to the claude / codex / pi / hive endpoints of intent_aligner.llm.
# Every message is an independent one-shot call: no history is carried over.
#
# Usage: scripts/llm-chat.sh [--home DIR] [claude|codex|pi|hive]
# Set PY to the Python that has intent_aligner installed. --home is passed on to
# intent_aligner.llm; without it the wrapper's default home is used.

set -euo pipefail

PY="${PY:-python3}"

declare -A DEFAULT_MODEL=(
    [claude]="opus"
    [codex]="gpt-6-astra"
    [pi]="fireworks/accounts/fireworks/models/glm-5p3-flash"
    [hive]="deployment/pc-qwen38-np8"
)

home_args=()
if [[ "${1:-}" == "--home" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "--home needs a directory." >&2
        exit 2
    fi
    home_args=(--home "$2")
    shift 2
fi

backend="${1:-claude}"
if [[ -z "${DEFAULT_MODEL[$backend]+set}" ]]; then
    echo "Unknown backend '$backend'; choose claude, codex, pi or hive." >&2
    exit 2
fi
model="${DEFAULT_MODEL[$backend]}"
effort="xhigh"
system="You are a helpful assistant."

system_file="$(mktemp)"
trap 'rm -f "$system_file"' EXIT

show() {
    echo "backend=$backend  model=$model  effort=$effort  home=${home_args[1]:-default}"
    echo "system: $system"
}

usage() {
    cat <<'EOF'
Type a message and press Enter to send it. Commands:
  /backend claude|codex|pi|hive
                             switch endpoint (model resets to that backend's default)
  /model NAME                set the model
  /effort low|medium|high|xhigh
  /system TEXT               set the system prompt
  /show                      print the current settings
  /help                      print this help
  /quit                      exit
EOF
}

send() {
    printf '%s' "$system" > "$system_file"
    # The wrapper prints the reply on stdout and the transcript path on stderr.
    if ! printf '%s' "$1" | "$PY" -m intent_aligner.llm \
        --backend "$backend" --model "$model" --effort "$effort" --system "$system_file" \
        "${home_args[@]}"; then
        echo "[call failed]" >&2
    fi
}

usage
show
while IFS= read -r -e -p "[$backend] > " line; do
    [[ -n "$line" ]] && history -s "$line"
    case "$line" in
        "") ;;
        /quit) break ;;
        /help) usage ;;
        /show) show ;;
        /backend\ *)
            next="${line#/backend }"
            if [[ -n "${DEFAULT_MODEL[$next]+set}" ]]; then
                backend="$next"
                model="${DEFAULT_MODEL[$backend]}"
                show
            else
                echo "Unknown backend '$next'; choose claude, codex, pi or hive."
            fi
            ;;
        /model\ *) model="${line#/model }"; show ;;
        /effort\ *)
            next="${line#/effort }"
            case "$next" in
                low | medium | high | xhigh) effort="$next"; show ;;
                *) echo "Unknown effort '$next'; choose low, medium, high or xhigh." ;;
            esac
            ;;
        /system\ *) system="${line#/system }"; show ;;
        /*) echo "Unknown command '$line'; /help lists the commands." ;;
        *) send "$line" ;;
    esac
done
