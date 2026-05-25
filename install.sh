#!/usr/bin/env sh
set -eu

LANGUAGE="${BACKBONE2RWKV_LANG:-zh}"
REQUESTED_SKILL="${BACKBONE2RWKV_SKILL:-backbone2rwkv}"
REF="${BACKBONE2RWKV_REF:-main}"
PROJECT_ROOT="${1:-$(pwd)}"
REPO="Jellyfish042/backbone2rwkv_skill"

case "$LANGUAGE" in
  zh|en) ;;
  *)
    echo "Unsupported language: $LANGUAGE" >&2
    echo "Use BACKBONE2RWKV_LANG=zh or BACKBONE2RWKV_LANG=en." >&2
    exit 1
    ;;
esac

case "$REQUESTED_SKILL" in
  backbone2rwkv)
    SOURCE_DIR="backbone2rwkv_$LANGUAGE/backbone2rwkv"
    DEST_NAME="backbone2rwkv"
    ;;
  optimize-rwkv7)
    SOURCE_DIR="optimize_rwkv7_$LANGUAGE/optimize-rwkv7"
    DEST_NAME="optimize-rwkv7"
    ;;
  *)
    echo "Unsupported skill: $REQUESTED_SKILL" >&2
    echo "Use BACKBONE2RWKV_SKILL=backbone2rwkv or BACKBONE2RWKV_SKILL=optimize-rwkv7." >&2
    exit 1
    ;;
esac

TMP_ROOT="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

ZIP_PATH="$TMP_ROOT/source.zip"
URL="https://github.com/$REPO/archive/refs/heads/$REF.zip"

echo "Downloading $REPO@$REF..."
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$URL" -o "$ZIP_PATH"
elif command -v wget >/dev/null 2>&1; then
  wget -q "$URL" -O "$ZIP_PATH"
else
  echo "curl or wget is required." >&2
  exit 1
fi

unzip -q "$ZIP_PATH" -d "$TMP_ROOT"
SOURCE_ROOT="$(find "$TMP_ROOT" -maxdepth 1 -type d -name 'backbone2rwkv_skill-*' | head -n 1)"
SOURCE_SKILL="$SOURCE_ROOT/$SOURCE_DIR"

if [ ! -f "$SOURCE_SKILL/SKILL.md" ]; then
  echo "Could not find skill at $SOURCE_SKILL." >&2
  exit 1
fi

SKILLS_DIR="$PROJECT_ROOT/.codex/skills"
DEST_SKILL="$SKILLS_DIR/$DEST_NAME"

mkdir -p "$SKILLS_DIR"
rm -rf "$DEST_SKILL"
cp -R "$SOURCE_SKILL" "$DEST_SKILL"

echo "Installed '$DEST_NAME' ($LANGUAGE) to:"
echo "  $DEST_SKILL"
