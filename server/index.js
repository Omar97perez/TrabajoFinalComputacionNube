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


const app = express();
app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())
app.use(cors())

//Conexión con la base de datos, cuando se despliegue en servidor  se tendrá que cambiar la dirección
mongoose.connect('mongodb://omar:antonio1997@cluster0-shard-00-00-svm5b.mongodb.net:27017,cluster0-shard-00-01-svm5b.mongodb.net:27017,cluster0-shard-00-02-svm5b.mongodb.net:27017/TrabajoFinalCN?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true')
  .then(db => console.log('DB conectada')) //Imprimir DB conectada en caso de que todo vaya bien
  .catch(err => console.error(err)); //Imprime error si no se puedo conectar

//Ajustes
  //Si el servidor tiene puerto lo coge sino pone el puerto 3000
  app.set('port', process.env.PORT || 3000  );

  //Sever escucha en el puerto x te lo muestra por pantalla
  app.listen(app.get('port'), () =>{
      console.log('Server on port', app.get('port'));
  });