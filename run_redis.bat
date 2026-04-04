@echo off
echo =======================================================
echo Starting Local Redis Server (Port 6379)...
echo This window needs to stay open while Celery runs.
echo =======================================================
cd redis
.\redis-server.exe
pause
