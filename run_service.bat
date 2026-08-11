@echo off
REM Launches the Seed-VC v2 voice-conversion service using its own Python 3.11
REM venv (seed-vc\venv\) - needs torch 2.4.0+cu124, a much newer stack than
REM backend_dlc's own fastapi/aiortc environment, so it cannot run in-process;
REM see PROJECT_HANDOFF.md for why. Invoked by RunWebAppDeepLiveCam.bat, or run
REM directly. Mirrors chatterbox\run_service.bat's pattern exactly - that one
REM still serves the original DeepFaceLive app (backend/) unchanged, on a
REM different port (8100 vs this one's 8101), so both can run side by side.
cd /D %~dp0
venv\Scripts\python.exe service.py --host 127.0.0.1 --port 8101
pause
