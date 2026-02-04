const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');
const path = require('path');

async function convertSvgToPng() {
    try {
        // Create output directory if it doesn't exist
        const outputDir = path.join(__dirname, '..', 'images');
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        // Paths
        const svgPath = path.join(outputDir, 'icon.svg');
        const pngPath = path.join(outputDir, 'icon.png');

        // Load SVG
        const img = await loadImage(svgPath);
        
        // Create canvas and draw image
        const canvas = createCanvas(1024, 1024);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 1024, 1024);
        
        // Save as PNG
        const buffer = canvas.toBuffer('image/png');
        fs.writeFileSync(pngPath, buffer);
        
        console.log(`Successfully converted ${svgPath} to ${pngPath}`);
    } catch (error) {
        console.error('Error converting SVG to PNG:', error);
    }
}

convertSvgToPng();
