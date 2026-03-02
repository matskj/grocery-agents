@echo off
setlocal

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "RUSTUP=%USERPROFILE%\.cargo\bin\rustup.exe"

if not exist "%VCVARS%" echo Error: vcvars64.bat not found at: %VCVARS% & echo Install Visual Studio Build Tools 2022 with C++ tools. & exit /b 1

if not exist "%RUSTUP%" echo Error: rustup.exe not found at: %RUSTUP% & echo Install Rust via rustup, then open a new terminal. & exit /b 1

call "%VCVARS%"
if errorlevel 1 echo Error: failed to initialize Visual C++ build environment. & exit /b 1

"%RUSTUP%" run stable-x86_64-pc-windows-msvc cargo %*
