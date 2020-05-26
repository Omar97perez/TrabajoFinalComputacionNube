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
                      <h3 class="modal-title" id="exampleModalLabel">Visualizar Imagen</h3>
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
                      <h3 class="modal-title" id="exampleModalLabel">Visualizar Imagen</h3>
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
              <h1 class="title-single">Algoritmos</h1>
            </div>
          </div>
        </div>

        <div class="card-deck mt-5">
          <div class="card" style="border: 1px solid #343a40!important">
            <img class="card-img-top" src="/img/ScikitLearn.png" alt="Card image cap">
            <div class="card-body">
              <h5 class="card-title">Scikit Learn</h5>
              <p class="card-text">Use la librería de Scikit Learn en la nube.</p>
              <div class="card-footer mt-2" align="center" style="position: relative!important; border-top: 1px solid  #343a40!important;">
                <button class="btn btn-dark">Acceder</button>
              </div>
            </div>
          </div>
          <div class="card" style="border: 1px solid #343a40!important">
            <img class="card-img-top" src="/img/OpenMP.jpg" alt="Card image cap">
            <div class="card-body">
              <h5 class="card-title">OpenMP</h5>
              <p class="card-text">Use la librería de Scikit Learn hybto con OpenMP en la nube.</p>
              <div class="card-footer mt-2" align="center" style="position: relative!important; border-top: 1px solid  #343a40!important;">
                <button class="btn btn-dark">Acceder</button>
              </div>
            </div>
          </div>
          <div class="card" style="border: 1px solid #343a40!important">
            <img class="card-img-top" src="/img/Spark.jpeg" alt="Card image cap">
            <div class="card-body">
              <h5 class="card-title">Spark</h5>
              <p class="card-text">Use Spark estadísctico en la nube.</p>
              <div class="card-footer mt-2" align="center" style="position: relative!important; border-top: 1px solid  #343a40!important;">
                <button class="btn btn-dark">Acceder</button>
              </div>
            </div>
          </div>
        </div>
        
      </div>
    </section>


  </div>
</template>


<script>

</script>