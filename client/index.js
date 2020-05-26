const express = require('express')

const app = express();
app.use(express.static('public'));

//Si el servidor tiene puerto lo coge sino pone el puerto 4100
app.set('port', process.env.PORT || 4100  );

//Ficheros estáticos, coge el index.html dentro de public
app.use(express.static(__dirname + '/public'));

//Sever escucha en el puerto x te lo muestra por pantalla
app.listen(app.get('port'), () =>{
    console.log('Client on port', app.get('port'));
});