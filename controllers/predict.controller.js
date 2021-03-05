const tf = require('@tensorflow/tfjs');
const automl = require('@tensorflow/tfjs-automl')
const image = require('get-image-data');
const fs = require('fs');

exports.makePredictions = async (req, res, next) => {
  const imagePath = './images/test-image.jpg';
  const MODEL_URL = "https://model-tf.s3.amazonaws.com/tf_js-demon_model-2021-03-04T105453.915998Z/model.json"
  try {
    const loadModel = async (img) => {
      const output = {};
      // laod model
      const model = await automl.loadImageClassification(MODEL_URL);
      // classify
      output.predictions = await model.classify(img);
      
      res.statusCode = 200;
      res.json(output);
    };
    image(imagePath, async (err, imageData) => {
      // pre-process image
      const numChannels = 3;
      const numPixels = imageData.width * imageData.height;
      const values = new Int32Array(numPixels * numChannels);
      const pixels = imageData.data;
      for (let i = 0; i < numPixels; i++) {
        for (let channel = 0; channel < numChannels; ++channel) {
          values[i * numChannels + channel] = pixels[i * 4 + channel];
        }
      }
      const outShape = [imageData.height, imageData.width, numChannels];
      const input = tf.tensor3d(values, outShape, 'int32');
      await loadModel(input);

      // delete image file
      fs.unlinkSync(imagePath, (error) => {
        if (error) {
          console.error(error);
        }
      });
    });
  } catch (error) {
    console.log(error)
  }
};