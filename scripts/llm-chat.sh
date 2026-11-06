#!/usr/bin/env bash
# Send single messages to the claude / codex / pi endpoints of intent_aligner.llm.
# Every message is an independent one-shot call: no history is carried over.
#
# Usage: scripts/llm-chat.sh [claude|codex|pi]
# Set PY to the Python that has intent_aligner installed.

set -euo pipefail

PY="${PY:-python3}"

declare -A DEFAULT_MODEL=(
    [claude]="opus"
    [codex]="gpt-6-astra"
    [pi]="fireworks/accounts/fireworks/models/glm-5p3-flash"
)

backend="${1:-claude}"
if [[ -z "${DEFAULT_MODEL[$backend]+set}" ]]; then
    echo "Unknown backend '$backend'; choose claude, codex or pi." >&2
    exit 2
fi
model="${DEFAULT_MODEL[$backend]}"
effort="xhigh"
system="You are a helpful assistant."

system_file="$(mktemp)"
trap 'rm -f "$system_file"' EXIT

show() {
    echo "backend=$backend  model=$model  effort=$effort"
    echo "system: $system"
}

usage() {
    cat <<'EOF'
Type a message and press Enter to send it. Commands:
  /backend claude|codex|pi   switch endpoint (model resets to that backend's default)
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
        --backend "$backend" --model "$model" --effort "$effort" --system "$system_file"; then
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
                echo "Unknown backend '$next'; choose claude, codex or pi."
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
