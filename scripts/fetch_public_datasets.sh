#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/raw}"
KO_DIR="$ROOT/korean-hate-speech"
EN_DIR="$ROOT/tweeteval"
KO_COMMIT="f8d05dce2b22007bb149e5139c0060c68ad8f94b"
EN_COMMIT="4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"

mkdir -p "$ROOT"

clone_sparse() {
  local url="$1"
  local destination="$2"
  local commit="$3"
  shift 3
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$url" "$destination"
  fi
  git -C "$destination" sparse-checkout init --cone
  git -C "$destination" sparse-checkout set "$@"
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

clone_sparse \
  https://github.com/kocohub/korean-hate-speech.git \
  "$KO_DIR" \
  "$KO_COMMIT" \
  labeled

clone_sparse \
  https://github.com/cardiffnlp/tweeteval.git \
  "$EN_DIR" \
  "$EN_COMMIT" \
  datasets/offensive

printf 'Korean commit:  %s\n' "$(git -C "$KO_DIR" rev-parse HEAD)"
printf 'TweetEval commit: %s\n' "$(git -C "$EN_DIR" rev-parse HEAD)"
