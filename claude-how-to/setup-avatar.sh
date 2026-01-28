#!/bin/bash

# Quick Setup Script for Kalidokit Avatar Integration

echo "🚀 Setting up Kalidokit + VRM Avatar for DuoSign..."

# Install dependencies
echo "📦 Installing npm packages..."
npm install kalidokit three @pixiv/three-vrm

# Create directories
echo "📁 Creating avatar directory..."
mkdir -p public/avatars

# Download sample avatar
echo "🤖 Downloading sample VRM avatar..."
curl -L https://github.com/vrm-c/vrm-specification/raw/master/samples/AliciaSolid.vrm \
  -o public/avatars/default-avatar.vrm

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy the AvatarRenderer.tsx component to src/components/app/"
echo "2. Copy the poseToKalidokit.ts utility to src/utils/"
echo "3. Replace SkeletonRenderer with AvatarRenderer in OutputPlayer.tsx"
echo "4. Click a gloss card to see your avatar animate!"
echo ""
echo "📚 See KALIDOKIT_INTEGRATION_GUIDE.md for full details"
