@echo off
cd /d "%~dp0"
C:\msys64\mingw64\bin\g++.exe -O2 -static -static-libgcc -static-libstdc++ -fopenmp -o p3a_engine.exe p3a_engine.cpp > _build_out.txt 2>&1
echo build exit=%errorlevel% >> _build_out.txt
type _build_out.txt
