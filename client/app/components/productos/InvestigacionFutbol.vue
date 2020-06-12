<template>
  <div>
      <!-- Modal Métodos -->
      <div class="modal fade" id="ModalEjecutarMetodo" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
          <div class="modal-dialog" role="document">
          <div class="modal-content">
              <div class="modal-header">
              <h3 class="modal-title" id="TitleMethod"></h3>
              <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                  <span aria-hidden="true">&times;</span>
              </button>
              </div>
              <div class="modal-body" align="center">
                  <div class="form-group" align="left">
                    <label class="font-weight-bold">Algortimo</label>
                    <select class="form-control" id="AlgoritmoSeleccionado">
                      <option value="1">Regresión Lineal</option>
                      <option value="2">Árbol de Regresión</option>
                      <option value="3">Regresión de Árbol Aleatorio</option>
                      <option value="4">Red Neuronal</option>
                    </select>
                  </div>
                  <form id="ModalMetodo"></form>
                  <form class="mt-3" align="left">
                      <label class="font-weight-bold">Archivo</label>
                  </form>
                  <input type="file" id="fileExecuteMethod">
              </div>
                  <div class="modal-footer">
                  <button type="button" class="btn btn-dark" @click="DescargarFicheroPrueba()">Descargar Archivo Prueba</button>
                  <button type="button" class="btn btn-primary" data-dismiss="modal" @click="EjecutarAlgoritmo()">Enviar</button>
                  <button type="button" class="btn btn-danger" data-dismiss="modal">Cerrar</button>
              </div>
          </div>
          </div>
      </div>

      <!-- Modal Cargar Imagen -->
      <div class="modal fade bd-example-modal-lg" id="ModalCargaImagen" role="dialog" aria-labelledby="exampleModalLabel">
          <div class="modal-dialog modal-lg" role="document">
              <div class="modal-content">
                  <div class="modal-header">
                      <h3 class="modal-title" id="exampleModalLabel">Visualizar Resultado</h3>
                      <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                          <span aria-hidden="true">&times;</span>
                      </button>
                  </div>
                  <div class="modal-body" >
                      <div class="d-flex justify-content-center">
                          <div class="spinner-border" role="status">
                              <span class="sr-only">Loading...</span>
                          </div>
                      </div>
                  </div>
                  <div class="modal-footer">
                      <button type="button" class="btn btn-danger" data-dismiss="modal">Cerrar</button>
                  </div>
              </div>
          </div>
      </div>

      <!-- Modal Ver Imágenes -->
      <div class="modal fade bd-example-modal-lg" id="ModalVerImagen" role="dialog" aria-labelledby="exampleModalLabel">
          <div class="modal-dialog modal-lg" role="document">
              <div class="modal-content">
                  <div class="modal-header">
                      <h3 class="modal-title" id="exampleModalLabel">Visualizar Resultado</h3>
                      <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                          <span aria-hidden="true">&times;</span>
                      </button>
                  </div>
                  <div class="modal-body" align="center">
                      <img class="img-fluid mt-3" id="myimage"/>
                  </div>
                  <div class="modal-footer">
                      <button type="button" class="btn btn-primary" id="prueba" rel="myimage" @click="DescargarImagen()">Descargar</button>
                      <button type="button" class="btn btn-danger" data-dismiss="modal">Cerrar</button>
                  </div>
              </div>
          </div>
      </div>

    <section class="intro-single">
      <div class="container">
        <div class="row">
          <div class="col-md-12 col-lg-8">
            <div class="title-single-box">
              <h1 class="title-single">Investigación de Fútbol</h1>
            </div>
          </div>
        </div>      
        <div class="mt-5 font-weight-bold" align="center">
          <h1>Introducción</h1>
        </div>
        <div class="mt-5" >
          <p style="font-size:20px;">
            Este proyecto consiste en un estudio sobre cómo podremos predecir la evolución de un equipo de fútbol en su liga. Para poder realizar esta acción, es necesario disponer de algunos datos de la liga actual (resultados de los últimos partidos,..).
          </p> 
          <p style="font-size:20px;">
            En este proyecto, también podremos ver como iremos probando con diferentes algoritmos de machine learning para intentar encontrar el algoritmo perfecto para este tipo de problema. De igual forma, estos se irán subiendo al proyecto realizado en la asignatura denominada “Computación en la Nube” en el Framework denominado “GraphCloud”.
          </p>
          <p style="font-size:20px;">
            Por último, cabe resaltar que se intentará sacar unas conclusiones finales sobre el estudio para que nos sirva para poder aplicarlas a un equipo de fútbol en concreto (si se desease).
          </p>
        </div>

        <div class="mt-5 font-weight-bold" align="center">
          <h1>Ejecución</h1>
        </div> 

        <div class="row">
          <div class="col-1"></div>
          <div class="card col-md-4 col-s-12 mt-5" style="border: 1px solid #343a40!important">
            <img class="card-img-top" src="/img/futbol.jpeg" alt="Card image cap">
            <div class="card-body">
              <h5 class="card-title">Ejecutar previsión de fútbol</h5>
              <p class="card-text">Esta opción nos permite poder predecir el puesto que va a logarr un equipo sabiendo los 10 primeros partidos jugados.</p>
              <div class="card-footer mt-2" align="center" style="position: relative!important; border-top: 1px solid  #343a40!important;">
                <button class="btn btn-dark" data-toggle="modal" data-target="#ModalEjecutarMetodo" @click="CargarFormulario('ADM2', 'Estimar valor',2)">Acceder</button>
              </div>
            </div>
          </div>

          <div class="col-2"></div>

          <div class="card col-md-4 col-s-12 mt-5" style="border: 1px solid #343a40!important">
            <img class="card-img-top" src="/img/Excel.jpeg" alt="Card image cap">
            <div class="card-body">
              <h5 class="card-title">Ejecutar con otra fuente de datos</h5>
              <p class="card-text">Esta opción nos permite poder ejecutar un algoritmo de predicción para una fuente propia.</p>
              <div class="card-footer mt-2" align="center" style="position: relative!important; border-top: 1px solid  #343a40!important;">
                  <button class="btn btn-dark" data-toggle="modal" data-target="#ModalEjecutarMetodo" @click="CargarFormulario('ADM', 'Estimar valor',2)">Acceder</button>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  </div>
</template>


<script>

import axios from 'axios';

class Buscador {
  constructor(busqueda = '',tipo = '',ciudad = '') {
    this.busqueda = busqueda;
  }
}
export default {
  name: 'clases',
  data() {
    return {
      Productos: [],
      Buscador: new Buscador(),
      busqueda: '',
      ciudad: '',
      tipo:'clases',
      titleMethod:'',
      Paginacion: [],
      elemento:''
    }
  },
  created() {
    this.getProductos();
  },
  methods: {
    getProductos() {
      fetch('/api/Get/Methods')
        .then(res => res.json())
        .then(data => {
          this.Paginacion = data["Methods"];
          this.Productos = this.Paginacion.slice(0,this.tampagina);
        });
    },
    CargarFormulario(title, titleModal, elemento){
      this.elemento = elemento;
      this.titleMethod = title;
      var position = this.Paginacion.findIndex(method => method.Name === title);
      document.getElementById('TitleMethod').innerHTML = titleModal;
      document.getElementById('ModalMetodo').innerHTML = "";
      for (var x=1;x<(Object.keys(this.Paginacion[position].Elements).length);x++) { 
          document.getElementById('ModalMetodo').innerHTML += '<div class="mt-3 form-group" align="left"><label class="font-weight-bold" for="exampleInputEmail1">'+ this.Paginacion[position].Elements[x].NameInput + '</label><div class="input-group mb-3"><input type="text" name="' + this.Paginacion[position].Elements[x].Name + '" id="' + this.Paginacion[position].Elements[x].Name + "2" + '" value="' + this.Paginacion[position].Elements[x].value + '" class="form-control"> <div class="input-group-append"><button class="btn btn-outline-primary" type="button" data-toggle="collapse" href="#'+ this.Paginacion[position].Elements[x].Name +'" role="button" aria-expanded="false" aria-controls="collapseExample"><i class="fa fa-question-circle" aria-hidden="true"></i></button></div></div><small id="emailHelp" class="form-text text-muted">'+ this.Paginacion[position].Elements[x].DescriptionShort +'</small></div> <div class="collapse" id="' + this.Paginacion[position].Elements[x].Name +'"><div class="card card-body">'+ this.Paginacion[position].Elements[x].DescriptionLong +'</div></div>';
      }
    },
    Endrefresh()
    {
        document.getElementById("loading").className = "";
        location.reload(true);
    },
    EjecutarAlgoritmo(){
      $("#ModalCargaImagen").modal();
      var position = this.Paginacion.findIndex(method => method.Name === this.titleMethod);
      var elements = "";
      this.elemento = document.getElementById("AlgoritmoSeleccionado").value;
      elements += this.elemento;

      for (var x=1;x<(Object.keys(this.Paginacion[position].Elements).length);x++) { 
          elements +=  "-";
          elements += document.getElementById(this.Paginacion[position].Elements[x]["Name"] + "2").value;
      }

      var urlPostMetodo = "";

      if(elements == ""){
          urlPostMetodo = '/api/Execute/Algorithm/' + this.titleMethod + '/' + "no";
      }
      else{
          urlPostMetodo = '/api/Execute/Algorithm/' + this.titleMethod + '/' + elements;
      }
      var formData = new FormData();
      formData.append("file", document.getElementById("fileExecuteMethod").files[0]);


      $.ajax({
          url: urlPostMetodo,
          type: "POST",
          data: formData,
          processData: false,
          contentType: false,
          success: function(response) {
              $("#ModalCargaImagen").modal('hide');
              $('body').removeClass('modal-open');
              $('.modal-backdrop').remove();
              response = response.split(".");
              document.getElementById('myimage').src = '/api/Get/file/' + response[0] + ".png";
              $('body').removeClass('ModalCargaImagen');
              $("#ModalVerImagen").modal();
          },
          error: function(jqXHR, textStatus, errorMessage) {
              console.log(errorMessage); 
          }
      });
    },
    DescargarImagen() {
        var nameHC = document.getElementById('myimage');
        var link = document.createElement("a");
        link.download = "image.png";
        link.href = nameHC.src;
        link.click();
    },
    DescargarFicheroPrueba() 
    {
      var link = document.createElement("a");
      var position = this.Paginacion.findIndex(method => method.Name === this.titleMethod);
      link.download = this.Paginacion[position].file;
      link.href = "/api/Get/file/" + this.Paginacion[position].file;
      link.click();
    }
  },
};
</script>