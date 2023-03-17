# ChaoticNDETools

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://maximilian-gelbrecht.github.io/ChaoticNDETools.jl/dev/)
[![Build Status](https://github.com/maximilian-gelbrecht/ChaoticNDETools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/maximilian-gelbrecht/ChaoticNDETools.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package contains a collection of tools, functions and models that I developed in my work with Neural Differential Equations. The documentation lists all avaiable tools. The core functionality is providing a model `ChaoticNDE` that will assist setting up Neural Differential Equations with the new explicit parameter system of Flux. This is meant to be used in conjuction with another helper package [NODEData.jl](https://github.com/maximilian-gelbrecht/NODEData.jl) that provides a dataloader for sequence data. 

The repository also contains example scripts that train Neural (Partial) Differential Equations for various example dynamical systems in the `scripts` folder. 

