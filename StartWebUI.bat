@echo off
setlocal
chcp 65001 >nul 2>&1

pushd "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
set "APP=%~dp0st.py"

if not exist "%APP%" (
    echo [ERROR] st.py was not found.
    popd
    pause
    exit /b 1
)

if not exist "%PYTHON%" (
    echo [ERROR] The project virtual environment was not found:
    echo         %PYTHON%
    echo.
    echo Run "python setup_env.py" first, then double-click this file again.
    popd
    pause
    exit /b 1
)

echo Starting VideoLingo Web UI...
echo Open http://127.0.0.1:8501 if the browser does not open automatically.
echo Press Ctrl+C in this window to stop the Web UI.
echo.

set "PYTHONWARNINGS=ignore"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
"%PYTHON%" -m streamlit run "%APP%" --server.address 127.0.0.1 --server.port 8501 --server.headless false %*
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Web UI exited with code %EXIT_CODE%.
)

popd
pause
exit /b %EXIT_CODE%
