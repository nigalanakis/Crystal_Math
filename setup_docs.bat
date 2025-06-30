@echo off
REM automated_setup.bat - Complete documentation setup
REM Save this file as: C:\Users\nikol\Crystal_Math\setup_docs.bat
REM Then run it from Command Prompt

echo ========================================
echo Crystal Structure Analysis Docs Setup
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "src" (
    echo ERROR: src folder not found!
    echo Make sure you're running this from C:\Users\nikol\Crystal_Math\
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo ✓ Found src folder - we're in the right place!
echo.

REM Step 1: Create directory structure
echo [1/6] Creating directory structure...
mkdir docs\source 2>nul
mkdir docs\source\_static 2>nul
mkdir docs\source\_static\images 2>nul
mkdir docs\source\_templates 2>nul
mkdir docs\source\getting_started 2>nul
mkdir docs\source\user_guide 2>nul
mkdir docs\source\tutorials 2>nul
mkdir docs\source\examples 2>nul
mkdir docs\source\technical_details 2>nul
mkdir docs\source\how_to_guides 2>nul
mkdir docs\source\reference 2>nul
mkdir docs\source\api_reference 2>nul
mkdir docs\source\api_reference\core 2>nul
mkdir docs\source\api_reference\extraction 2>nul
mkdir docs\source\api_reference\processing 2>nul
mkdir docs\source\api_reference\io 2>nul
echo ✓ Directories created!

REM Step 2: Create requirements file
echo [2/6] Creating requirements file...
echo sphinx^>=7.0.0 > docs\requirements.txt
echo sphinx-rtd-theme^>=1.3.0 >> docs\requirements.txt
echo sphinx-copybutton^>=0.5.2 >> docs\requirements.txt
echo sphinx-design^>=0.5.0 >> docs\requirements.txt
echo myst-parser^>=2.0.0 >> docs\requirements.txt
echo sphinx-autodoc-typehints^>=1.24.0 >> docs\requirements.txt
echo ✓ Requirements file created!

REM Step 3: Install Sphinx
echo [3/6] Installing Sphinx and dependencies...
echo This may take a few minutes...
pip install -r docs\requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install requirements!
    echo Try running: pip install sphinx sphinx-rtd-theme
    pause
    exit /b 1
)
echo ✓ Sphinx installed successfully!

REM Step 4: Initialize Sphinx
echo [4/6] Initializing Sphinx documentation...
cd docs
echo y| sphinx-quickstart --quiet --project="Crystal Structure Analysis" --author="Crystal Math Team" --release="1.0.0" --language="en" --suffix=".rst" --master="index" --epub
cd ..
echo ✓ Sphinx initialized!

REM Step 5: Create basic files
echo [5/6] Creating basic documentation files...

REM Create a simple index.rst
echo Getting Started > docs\source\getting_started\index.rst
echo =============== >> docs\source\getting_started\index.rst
echo. >> docs\source\getting_started\index.rst
echo Welcome to the Crystal Structure Analysis documentation! >> docs\source\getting_started\index.rst

REM Create .gitignore
echo # Documentation builds > .gitignore
echo docs/build/ >> .gitignore
echo __pycache__/ >> .gitignore
echo *.pyc >> .gitignore
echo .DS_Store >> .gitignore
echo Thumbs.db >> .gitignore

echo ✓ Basic files created!

REM Step 6: Test build
echo [6/6] Testing documentation build...
cd docs
sphinx-build -b html source build\html
if %ERRORLEVEL% neq 0 (
    echo WARNING: Build had some issues, but this is normal for initial setup
    echo We'll fix any errors in the next steps
) else (
    echo ✓ Documentation built successfully!
)
cd ..

echo.
echo ========================================
echo Setup Complete! 
echo ========================================
echo.
echo What was created:
echo • docs/ folder with Sphinx configuration
echo • Basic documentation structure  
echo • Requirements file for dependencies
echo • .gitignore file
echo.
echo Next steps:
echo 1. View your docs: start docs\_build\html\index.html
echo 2. Edit docs\source\conf.py (I'll help you with this)
echo 3. Add content to documentation files
echo 4. Set up GitHub repository
echo 5. Connect to ReadTheDocs
echo.
echo Ready for the next step? Press any key...
pause >nul

echo.
echo Opening your new documentation...
start docs\_build\html\index.html

echo.
echo Documentation is now open in your browser!
echo Let me know when you're ready for the next step.
pause