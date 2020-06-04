require('rootpath')()
const express = require('express')
const morgan = require('morgan')
const mongoose = require('mongoose')
const cors = require('cors')
const bodyParser = require('body-parser')
const path = require('path');
const multer = require('multer');
let fs = require('fs');
var request = require('request');
const axios = require('axios')

const { KubeConfig } = require('kubernetes-client');
const Client = require('kubernetes-client').Client;

const kubeconfig = new KubeConfig();
kubeconfig.loadFromFile('./kubeconfig');

const crd = require('./Algoritmos/Spark/crd.json');

const Request = require('kubernetes-client/backends/request');

const app = express();
app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())
app.use(cors())

function executeMake(nameMethod) {
  require('child_process').execSync("sudo make -C ./Algoritmos/" + nameMethod + " all");
}

// Permite Subir Imagenes
let storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './Archivos')
  },
  filename: (req, file, cb) => {
    cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage });

//Conexión con la base de datos, cuando se despliegue en servidor  se tendrá que cambiar la dirección
mongoose.connect('mongodb://omar:antonio1997@cluster0-shard-00-00-svm5b.mongodb.net:27017,cluster0-shard-00-01-svm5b.mongodb.net:27017,cluster0-shard-00-02-svm5b.mongodb.net:27017/TrabajoFinalCN?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true')
  .then(db => console.log('DB conectada')) //Imprimir DB conectada en caso de que todo vaya bien
  .catch(err => console.error(err)); //Imprime error si no se puedo conectar

//Ajustes
//Si el servidor tiene puerto lo coge sino pone el puerto 3000
app.set('port', process.env.PORT || 3000);

//Sever escucha en el puerto x te lo muestra por pantalla
app.listen(app.get('port'), () => {
  console.log('Server on port', app.get('port'));
});

// Permite Ejecutar Métodos 
app.post('/api/Execute/Algorithm/:name/:Elements', upload.single('file'), (req, res) => {

  var elementsUrl = req.params.Elements.split("-");
  
  if ( req.file ) {
    var fileExit = req.file.filename.split(".");
  }

  fs.readFile("./Metodos.json", 'utf-8', (err2, data) => {
    var methods = JSON.parse(data);
    var element = methods['Methods'].findIndex(method => method.Name === req.params.name);
    var method = methods['Methods'][element];
    //executeMake(method["Name"]);
    var elements = method["Elements"];
    var stringFinal = "";

    if (elementsUrl.length != 1) {
      for (x = 0; x < elementsUrl.length; x++) {
        stringFinal += " " + elements[x]["Name"] + "=" + elementsUrl[x];
      }
    }

    if (req.params.name === "Spark") {

      
      const backend = new Request({ kubeconfig })
      const client = new Client({ backend, version: '1.13' })

      console.log(backend)
      
      try {
        const deploymentManifest = require('./Algoritmos/' + req.params.name + "/sparkpi.json")

        client.addCustomResourceDefinition(crd);

        console.log(client)

        client.apis['sparkoperator.k8s.io'].v1beta2.namespaces('default').sparkapplication.post({ body: deploymentManifest }).then((create) => {
          console.log('Create:', create);
          res.send(create);
        })
      
      } catch (err) {
        if (err.code !== 409) throw err
        client.apis['sparkoperator.k8s.io'].v1beta2.namespaces('default').deployments('spark-pi-again').put({ body: deploymentManifest }).then((create) => {
          console.log('Create:', create);
          res.send(create);
        })
      }

    } else {
      
      console.log("make -C ./Algoritmos/" + req.params.name + " file=../../Archivos/" + req.file.filename + " fileExit=../../Archivos/" + fileExit[0] + ".png " + stringFinal + " run");


      const exec = require('child_process').exec;
      exec("make -C ./Algoritmos/" + req.params.name + " file=../../Archivos/" + req.file.filename + " fileExit=../../Archivos/" + fileExit[0] + ".png " + stringFinal + " run", (err, stdout, stderr) => {
        res.send(req.file.filename);
      });
    }
  });
});

// Permite recoger Imágenes 
app.get('/api/Get/file/:name', (req, res) => {
  res.sendFile('./Archivos/' + req.params.name, { root: __dirname });
});


// Permite devolver El archivo con todos los Métodos
app.get('/api/Get/Methods', function (req, res) {
  res.sendFile('./Metodos.json', { root: __dirname });
});