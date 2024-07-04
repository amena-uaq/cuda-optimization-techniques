# Demostración de técnicas de optimización en CUDA

## Descripción

En este repositorio se encuentra un archivo principal en el cual se demuestra el uso del código en Matrix.cuh y el encabezado Matrix.cuh, en este último se encuentran multiples implementaciones para la creación de una matriz, cada una acumulando las optimizaciones previas.

## Consideraciones

El código proporcionado no cuenta con código para un build system, sin embargo, dado que no se utilizaron librerias externas y que el codigo proporcionado consta de dos archivos, teóricamente el programa puede ser compilado manualmente haciendo uso de nvcc.

## Hardware utilizado 

Procesador AMD Ryzen 5 5600x a 4.2GHZ, 48GB de ram DDR4 a 3200 MHZ, GPU Nvidia RTX 3050 con 8GB de VRAM, NVME de 1TB con una lectura y escritura de 3500 MB/s, fuente de poder de 750 Watts Bronce.

## Software utilizado
- Windows 11 Pro
- Visual Studio 2022
- CUDA SDK 12.3
