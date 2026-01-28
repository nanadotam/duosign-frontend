#!/bin/bash

# Quick VRM Verification Script

echo "🔍 Checking DuoSign-Proto.vrm setup..."
echo ""

# Check if file exists
if [ -f "public/avatars/DuoSign-Proto.vrm" ]; then
    echo "✅ VRM file found in public/avatars/"
    
    # Check file size
    size=$(du -h "public/avatars/DuoSign-Proto.vrm" | cut -f1)
    echo "📦 File size: $size"
    
    # Check if it's a valid GLB/VRM (they start with 'glTF')
    if head -c 4 "public/avatars/DuoSign-Proto.vrm" | grep -q "glTF"; then
        echo "✅ Valid VRM/GLB format detected"
    else
        echo "⚠️  Warning: File might not be valid VRM format"
    fi
    
else
    echo "❌ VRM file not found!"
    echo ""
    echo "Please place your DuoSign-Proto.vrm file in:"
    echo "  public/avatars/DuoSign-Proto.vrm"
    echo ""
    echo "Create the directory if needed:"
    echo "  mkdir -p public/avatars"
    echo "  mv DuoSign-Proto.vrm public/avatars/"
    exit 1
fi

echo ""
echo "🔧 Checking npm dependencies..."

# Check if packages are installed
if npm list kalidokit three @pixiv/three-vrm &> /dev/null; then
    echo "✅ All required packages installed"
else
    echo "⚠️  Missing dependencies. Run:"
    echo "  npm install kalidokit three @pixiv/three-vrm"
fi

echo ""
echo "📁 Checking project structure..."

# Check for required directories
if [ -d "src/components/app" ]; then
    echo "✅ src/components/app/ exists"
else
    echo "⚠️  src/components/app/ not found"
fi

if [ -d "src/utils" ]; then
    echo "✅ src/utils/ exists"
else
    echo "⚠️  src/utils/ not found - create it with: mkdir -p src/utils"
fi

echo ""
echo "✨ Next steps:"
echo "1. Copy AvatarRenderer.tsx to src/components/app/"
echo "2. Copy poseToKalidokit.ts to src/utils/"
echo "3. Update OutputPlayer.tsx to use AvatarRenderer"
echo "4. Run 'npm run dev' and click a gloss card!"
