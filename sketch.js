let camera;
var buttonEntrenar;
var texto;
var modelo;
var knn;
var clasificando = false;
var inputTexBox;
var buttonTextBox;
var buttonGuardar;
var buttonCargar;

function setup() {
    createCanvas(320, 240);
    background(255, 0, 0);
    camera = createCapture(VIDEO);
    camera.size(320, 240);
    camera.hide();

    modelo = ml5.featureExtractor('MobileNet', ModeloListo);
    knn = ml5.KNNClassifier();

    createP("Presiona botones para entrenar");

    var buttonArduino = createButton('Arduino');
    buttonArduino.class("BotonEntrenar");

    var buttonRedboard = createButton('Redboard');
    buttonRedboard.class("BotonEntrenar");

    var buttonESP8266 = createButton('ESP8266');
    buttonESP8266.class("BotonEntrenar");

    var buttonESP32 = createButton('ESP32');
    buttonESP32.class("BotonEntrenar");

    var buttonNada = createButton('Nada');
    buttonNada.class("BotonEntrenar");

    createP("Entrena usando TextBox");
    inputTexBox = createInput("Cosa 2");
    buttonTextBox = createButton("Entrenar con " + inputTexBox.value());
    buttonTextBox.mousePressed(entrenarTextBox);

    createP("Guarda o Carga tu Neurona");
    buttonGuardar = createButton("Guardar");
    buttonCargar = createButton("Cargar");
    buttonGuardar.mousePressed(guardarNeurona);
    buttonCargar.mousePressed(cargarNeurona);



    texto = createP("Modelo no Listo, esperando");

    buttonEntrenar = selectAll(".BotonEntrenar");

    for (let index = 0; index < buttonEntrenar.length; index++) {
        let element = buttonEntrenar[index];
        element.style("margin", '5px');
        element.style("padding", '6px');
        element.mousePressed(presionandoButton);
    }
}

function draw() {
    //ellipse(50, 50, 80, 80);
    image(camera, 0, 0, 320, 240);
    buttonTextBox.html("Entrenar con " + inputTexBox.value());

    if (knn.getNumLabels() > 0 && !clasificando) {
        //clasificar();
        setInterval(clasificar, 500);
        clasificando = true;
    }
}

function presionandoButton() {
    var nameButton = this.elt.innerHTML;
    console.log("Entrenando con " + nameButton);
    entrenarKNN(nameButton);
}

function ModeloListo() {
    console.log("Modelo Listo");
    texto.html("Modelo Listo, empieza a entrenar!");
}

function entrenarKNN(ObjEntrenar) {
    let image = modelo.infer(camera); //numero es la imagen
    knn.addExample(image, ObjEntrenar); //agrega al objentrenar
}

function clasificar() {
    let imagen = modelo.infer(camera);
    knn.classify(imagen, function(error, result) {
        if (error) {
            console.error();
        } else {
            if ((result.label).charCodeAt(0) >= 48 && (result.label).charCodeAt(0) <= 57) {
                let txt = Object.keys(result.confidencesByLabel)[result.confidences[parseInt(result.label)]];
                texto.html("Es un " + txt);
                console.log(txt);
            } else {
                let txt = result.label;
                texto.html("Es un " + txt);
                console.log(txt);
            }
        }
    });
}

function entrenarTextBox() {
    let image = modelo.infer(camera); //numero es la imagen
    knn.addExample(image, inputTexBox.value()); //agrega al objentrenar
}

function guardarNeurona() {
    console.log("Cargando una Neurona");
    if (clasificando) {
        save(knn, "modelo.json");
    }
}

function cargarNeurona() {
    console.log("Cargando una Neurona");
    knn.load('./modelo.json', function() {
        console.log("Neurona Cargada");
        texto.html("Neurona cargada de archivo");
    });
}


// Temporary save code until ml5 version 0.2.2
const save = (knn, name) => {
    const dataset = knn.knnClassifier.getClassifierDataset();
    if (knn.mapStringToIndex.length > 0) {
        Object.keys(dataset).forEach(key => {
            if (knn.mapStringToIndex[key]) {
                dataset[key].label = knn.mapStringToIndex[key];
            }
        });
    }
    const tensors = Object.keys(dataset).map(key => {
        const t = dataset[key];
        if (t) {
            return t.dataSync();
        }
        return null;
    });
    let fileName = 'myKNN.json';
    if (name) {
        fileName = name.endsWith('.json') ? name : `${name}.json`;
    }
    saveFile(fileName, JSON.stringify({
        dataset,
        tensors
    }));
};

const saveFile = (name, data) => {
    const downloadElt = document.createElement('a');
    const blob = new Blob([data], {
        type: 'octet/stream'
    });
    const url = URL.createObjectURL(blob);
    downloadElt.setAttribute('href', url);
    downloadElt.setAttribute('download', name);
    downloadElt.style.display = 'none';
    document.body.appendChild(downloadElt);
    downloadElt.click();
    document.body.removeChild(downloadElt);
    URL.revokeObjectURL(url);
};

//broken privados y publicos
//https: //shiftr.io/
//http://mqtt.org/
//https://github.com/mqtt/mqtt.github.io/wiki/libraries