const canvas = document.getElementById('main-canvas');
const smallCanvas = document.getElementById('small-canvas');
const displayBox = document.getElementById('prediction');
const display1 = document.getElementById('acc');

const inputBox = canvas.getContext('2d');
const smBox = smallCanvas.getContext('2d');

const ar=['0', '1', '2','3', '4', '5', '6', '7', '8', '9', 'A', 'B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R','S', 'T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i', 'j','k', 'l', 'm','n','o', 'p','q','r','s', 't','u', 'v', 'w', 'x','y','z'];
let isDrawing = false;
let model;

/* Loads trained model */
async function init() {
  model = await tf.loadModel('/home/ashi/Desktop/CNN/English/model_characters/model.json');
}

canvas.addEventListener('mousedown', event => {
  isDrawing = true;

  inputBox.strokeStyle = 'white';
  inputBox.lineWidth = '15';
  inputBox.lineJoin = inputBox.lineCap = 'round';
  inputBox.beginPath();
});

canvas.addEventListener('mousemove', event => {
  if (isDrawing) drawStroke(event.clientX, event.clientY);
});

canvas.addEventListener('mouseup', event => {
  isDrawing = false;
  updateDisplay(predict());
});

/* Draws on canvas */
function drawStroke(clientX, clientY) {
  // get mouse coordinates on canvas
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;

  // draw
  inputBox.lineTo(x, y);
  inputBox.stroke();
  inputBox.moveTo(x, y);
}

/* Makes predictions */
function predict() {
  let values = getPixelData();
  let predictions = model.predict(values).dataSync();

  return predictions;
}

/* Returns pixel data from canvas after applying transformations */
function getPixelData() {
  smBox.drawImage(inputBox.canvas, 0, 0, smallCanvas.width, smallCanvas.height);
  const imgData = smBox.getImageData(0, 0, smallCanvas.width, smallCanvas.height);

  // preserve and normalize values from red channel only
  let values = [];
  for (let i = 0; i < imgData.data.length; i += 4) {
    values.push(imgData.data[i] / 255);
  }
  values = tf.reshape(values, [1, 28, 28, 1]);
  return values;
}

/* Displays predictions on screen */
function updateDisplay(predictions) {
  // Find index of best prediction, which corresponds to the predicted value
  const bestPred = predictions.indexOf(Math.max(...predictions));

  displayBox.innerText = ar[bestPred];
  const x = Math.floor((Math.max(...predictions)*100)*100)/100;
    display1.innerText = "Accuracy for the Predicted Output : "+x;
}

document.getElementById('erase').addEventListener('click', erase);

/* Clears canvas */
function erase() {
  inputBox.fillStyle = 'black';
  inputBox.fillRect(0, 0, canvas.width, canvas.height);
  displayBox.innerText = '';
}

erase();
init();
