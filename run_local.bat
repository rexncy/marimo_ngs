@echo off
setlocal
set "REPO_DIR=marimo_ngs"

if exist "%REPO_DIR%\" (
  pushd "%REPO_DIR%" && git pull && popd
) else (
  git clone https://github.com/rexncy/marimo_ngs.git || exit /b 1
)

pushd "%REPO_DIR%" || exit /b 1
uv run main.py %*
set "ERR=%ERRORLEVEL%"
popd
exit /b %ERR%