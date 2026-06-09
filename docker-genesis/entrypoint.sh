#!/bin/bash
# Entrypoint for the Genesis dev image.
# - Runs as the host user's UID/GID (passed via LOCAL_USER_ID/LOCAL_GROUP_ID) so
#   files written to the mounted checkout stay owned by the host user.
# - Editable-installs the mounted Genesis checkout the first time (idempotent).
set -e

GENESIS_DIR="${GENESIS_DIR:-/workspace/Uni-Genesis}"

install_genesis() {
    if [ ! -f "${GENESIS_DIR}/pyproject.toml" ]; then
        echo "[entrypoint] WARNING: no Genesis checkout at ${GENESIS_DIR} (mount it with -v)."
        return
    fi
    # `import genesis` gives a false positive when CWD == GENESIS_DIR (the source
    # tree is importable in-place). Check for a real install record instead.
    if python3 -m pip show genesis-world >/dev/null 2>&1; then
        return
    fi
    echo "[entrypoint] editable-installing Genesis from ${GENESIS_DIR} (user site) ..."
    # --user: system dist-packages isn't writable by the runtime user (~/.local is).
    # --no-deps: heavy deps are baked into the image.
    # --no-build-isolation: the build backend (setuptools/wheel/cython/numpy) is
    #   already installed; isolation would re-fetch them from PyPI and hang on this
    #   host's slow/contended network.
    python3 -m pip install --user --no-deps --no-build-isolation -e "${GENESIS_DIR}" >/dev/null
}

if [ -n "${LOCAL_USER_ID}" ]; then
    USER_NAME=$(getent passwd "${LOCAL_USER_ID}" | cut -d: -f1)
    if [ -z "${USER_NAME}" ]; then
        USER_NAME=genesis
        GROUP_ID="${LOCAL_GROUP_ID:-${LOCAL_USER_ID}}"
        getent group "${GROUP_ID}" >/dev/null 2>&1 || groupadd -g "${GROUP_ID}" "${USER_NAME}"
        useradd --shell /bin/bash --uid "${LOCAL_USER_ID}" --gid "${GROUP_ID}" -m "${USER_NAME}"
    fi
    # Run the editable install AS the target user (gosu), so --user lands in the
    # right ~/.local and file ownership is correct.
    export HOME="/home/${USER_NAME}"
    gosu "${USER_NAME}" env HOME="${HOME}" GENESIS_DIR="${GENESIS_DIR}" \
        bash -c "$(declare -f install_genesis); install_genesis"
    exec gosu "${USER_NAME}" "$@"
else
    install_genesis
    exec "$@"
fi
