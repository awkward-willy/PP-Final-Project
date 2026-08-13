

<h1 align="center">
  <br>
  <img src="https://hackedserver.org/choucroute/velociraptor.svg" alt="Simulant logo" width="256">
  <br>
</h1>


<h4 align="center">🐜 A C++ simulation of ant colonies.</h4>
</br>
<p align="center">
<img src="./ressources/animation.png" alt="Simulant animation" width="512">
</p>

## Features
- Support for multiple colonies
- Real time rendering with SDL2
- Optimization of the rendering with the graphic card
- Dynamic updating of entities and pheromones

## Code conventions

#### Docstrings
We followed javadocs guidelines with an additional decorator:
``@confidence 0-5``
- 0: it doesn't work
- 1: it "works" with known bugs
- 3: it works but can't be tested
- 4: it works and has been tested
- 5: it works, it has been tested and it is so simple that there is no doubt about it
