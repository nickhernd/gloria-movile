const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

async function convert() {
  console.log('Loading model...');
  const modelPath = path.resolve(__dirname, '../web_app/student_mobilenet_little.h5');
  const outputDir = path.resolve(__dirname, 'tfjs-model');

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const model = await tf.loadLayersModel(`file://${modelPath}`);
  await model.save(`file://${outputDir}`);
  console.log(`Model saved to ${outputDir}`);
}

convert();