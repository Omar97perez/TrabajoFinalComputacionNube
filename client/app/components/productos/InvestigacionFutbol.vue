<template>
  <div>
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

        <div class="row mt-5">
          <div class="col-1"></div>
          <div class="card col-4" style="border: 1px solid #343a40!important">
            <img class="card-img-top" src="/img/futbol.jpeg" alt="Card image cap">
            <div class="card-body">
              <h5 class="card-title">Ejecutar previsión de fútbol</h5>
              <p class="card-text">Esta opción nos permite poder predecir el puesto que va a logarr un equipo sabiendo los 10 primeros partidos jugados.</p>
              <div class="card-footer mt-2" align="center" style="position: relative!important; border-top: 1px solid  #343a40!important;">
                  <router-link :to="{ name: 'InvestigacionFutbol' }" >
                    <button class="btn btn-dark">Acceder</button>
                  </router-link>
              </div>
            </div>
          </div>

          <div class="col-2"></div>

          <div class="card col-4" style="border: 1px solid #343a40!important">
            <img class="card-img-top" src="/img/Excel.jpeg" alt="Card image cap">
            <div class="card-body">
              <h5 class="card-title">Ejecutar con otra fuente de datos</h5>
              <p class="card-text">Esta opción nos permite poder ejecutar un algoritmo de predicción para una fuente propia.</p>
              <div class="card-footer mt-2" align="center" style="position: relative!important; border-top: 1px solid  #343a40!important;">
                  <router-link :to="{ name: 'InvestigacionFutbol' }" >
                    <button class="btn btn-dark">Acceder</button>
                  </router-link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script></script>